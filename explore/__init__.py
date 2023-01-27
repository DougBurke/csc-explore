# Can we explore the CSC 2.0 "nicely"?
#

import glob
import logging
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from flask import Flask, flash, redirect, render_template, request, url_for

import pycrates

from sherpa.astro import io
from sherpa.astro.plot import DataPHAPlot
from sherpa.stats import Chi2Gehrels

from ciao_contrib.cda import csccli
from ciao_contrib.runtool import make_tool
from ciao_contrib.proptools import dates

from . import dbase


def create_app():

    # Currently there's no way to over-ride the settings.
    #
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'explore.sqlite'),
    )

    # logging.getLogger('sherpa').setLevel(logging.ERROR)
    logging.getLogger('sherpa').setLevel(logging.DEBUG)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def index():
        return redirect(url_for('home'))

    @app.route('/index.html', methods=('GET', 'POST'))
    def home():
        if request.method == 'POST':
            pos = request.form['position'].strip()
            if pos != '':
                db = dbase.get_db()
                ipath = app.instance_path
                ctr = search_loc(db, ipath, pos)
                if ctr is not None:
                    return redirect(f'/search/{ctr}')

                flash(f"No location found for '{pos}'")

            flash('No object was specified!')

        return render_template('home.html')

    @app.route('/searches')
    def searches():
        db = dbase.get_db()
        searches = db.execute(
            'SELECT counter,created,location,radius,nsrc'
            ' FROM searches'
            ' ORDER BY created DESC'
        ).fetchall()
        return render_template('searches.html', searches=searches)

    @app.route('/search/<counter>')
    def search(counter):
        db = dbase.get_db()
        search = db.execute(
            'SELECT created,location,radius,nsrc,filename'
            ' FROM searches'
            ' WHERE counter = ?'
            ' ORDER BY created DESC',
            (counter, )
        ).fetchall()

        if len(search) == 0:
            flash('No search found')
            return render_template('error.html'), 404

        if len(search) > 1:
            flash('Too many searches found! Does not compute...')
            return render_template('error.html'), 404

        table = read_csc_table(search[0]['filename'])

        # must be an easier way to do this
        names = table.get_column('name').values.copy()
        nset = set(names)
        idx = 1
        while len(nset) > 0:
            s = nset.pop()
            names[names == s] = idx
            idx += 1

        # we want the first bin to always be selected
        names = names.astype(np.int)
        diff = np.diff(names, prepend=0)
        idxs, = np.where(diff != 0)

        return render_template('search.html', search=search[0], table=table, idxs=idxs)

    @app.route('/source/<name>')
    def source(name):
        db = dbase.get_db()
        ipath = app.instance_path
        base = download_src(db, ipath, name)
        if base is None:
            flash(f'Somehow I do not know about {name}')
            return render_template('error.html'), 404

        parent = db.execute(
            'SELECT src_parent FROM sources WHERE src_name = ?',
            (name, )).fetchone()
        if parent is None:
            flash(f'Somehow I do not know about {name}')
            return render_template('error.html'), 404

        # This drops the NULL columns, which may be
        # confusing for the template.
        #
        tsvfile = db.execute('SELECT filename FROM searches WHERE counter = ?',
                             (parent['src_parent'], )).fetchone()
        cr = read_csc_table(tsvfile['filename'], alldata=True)

        # I should have this information stored somewhere rather
        # than being re-created each time.
        #
        idx = cr.get_column('name').values == name
        if not idx.any():
            flash(f'Somehow I know nothing about {name}')
            return render_template('error.html'), 404

        srccols = []
        obicols = ['Detection']
        store = srccols
        for col in cr.get_colnames():
            if col in ['name', 'sepn', 'detect_stack_id']:
                continue

            if col == 'obsid':
                store = obicols

            store.append(col)

        # The source-level data is independent of the obi-level
        # so we only need one row.
        #
        srcdata = []
        for col in srccols:
            coldata = cr.get_column(col).values[idx]
            srcdata.append(coldata[0])

        # Break up the obidata by stack.
        #
        stacks = cr.get_column('detect_stack_id').values[idx]
        obidata = {}
        for stack in set(stacks):
            obidata[stack] = []
            sidx = stacks == stack

            # create the "detection" (aka "tag") term and add it to
            # the start of the array
            tag = []
            for obsid, obi in zip(cr.get_column('obsid').values[idx][sidx],
                                  cr.get_column('obi').values[idx][sidx]):
                tag.append(f'{obsid:05d}_{obi:03d}')

            obidata[stack].append(tag)

            for col in obicols:
                if col == 'Detection':
                    continue

                coldata = cr.get_column(col).values[idx][sidx]
                obidata[stack].append(coldata)

        return render_template('source.html', name=name, stacks=set(stacks),
                               srcdata=srcdata, srccols=srccols,
                               obidata=obidata, obicols=obicols)

    @app.route('/source/<name>/<obistr>')
    def detection(name, obistr):

        try:
            obsid, obi = [int(i) for i in obistr.split('_')]
        except Exception as exc:
            print(f"DBG: {exc}")
            flash('Invalid URL')
            return render_template('error.html'), 404

        db = dbase.get_db()
        cur = db.execute('SELECT src_region_id,src_stack FROM sources WHERE src_name = ? AND src_obsid = ? AND src_obi = ?',
                         (name, obsid, obi)
        ).fetchone()
        if cur is None:
            print(f"NO MATCH: {name} {obsid} {obi}")
            flash('Invalid URL')
            return render_template('error.html'), 404

        stack = cur['src_stack']
        rid = f"{cur['src_region_id']:04d}"

        # argh - need to change from '2CXO J...' to '2CXOJ...'
        #
        fname = name.replace(' ', '')

        # just glob to get the files, assuming there's only one
        # version
        #
        path = app.instance_path
        indir = os.path.join(path, fname, obistr)
        files = {}
        allfiles = glob.glob(indir + '/*')
        for f in allfiles:
            if f.endswith('.svg'):
                continue

            base = os.path.basename(f)

            if not base.endswith('.fits.gz'):
                app.logger.warning(f'Skipping file: {f}')
                continue

            ftype = base[:-8].split('_')[-1]
            if ftype in files:
                app.logger.error(f'Multiple {ftype} in {indir}!')

            files[ftype] = f

        pat = os.path.join(app.instance_path, fname, stack,
                           f'{stack}*_r{rid}_reg3.fits*')
        allfiles = glob.glob(pat)
        for f in allfiles:
            # just over-write if we have multiple
            files['stksrcreg'] = f

        if len(files) == 0:
            flash(f'No data found for {name} / {obistr}')
            return render_template('error.html'), 404

        svgs = {}
        for ftype in ['lc3', 'pha3', 'regexp3', 'regevt3', 'psf3']:
            svgfile = make_svg(path, ftype, files)
            if svgfile is not None:
                svgs[ftype] = svgfile

        lc3 = read_svg(svgs['lc3'])
        pha3 = read_svg(svgs['pha3'])
        exp3 = read_svg(svgs['regexp3'])
        evt3 = read_svg(svgs['regevt3'])
        psf3 = read_svg(svgs['psf3'])
        return render_template('detection.html', name=name, obi=obistr,
                               lc3=lc3, pha3=pha3, exp3=exp3, evt3=evt3,
                               psf3=psf3)

    @app.errorhandler(404)
    def not_found(error):
        return render_template('error.html'), 404

    dbase.init_app(app)

    return app


def search_loc(db, path, loc, radius=1):
    """How many sources are near the location?

    Parameters
    ----------
    dbase
        The database object.
    path : str
        Location for the file.
    loc : str
        The name.
    radius : float, optional
        The search radius, in arcminutes.

    """

    tool = make_tool('search_csc')

    ctr = 1
    while True:
        out = os.path.join(path, f'results.{ctr}.tsv')
        if not os.path.exists(out):
            break

        ctr += 1

    try:
        tool(loc, radius=radius, radunit='arcmin', outfile=out)
    except OSError:
        # assume location not found
        return None

    cr = pycrates.read_file(f"{out}[opt kernel=text/tsv]")
    allnames = list(set(cr.get_column('name').values))
    nsrc = len(allnames)

    cur = db.execute(
        'INSERT INTO searches (location, radius, nsrc, filename) VALUES (?, ?, ?, ?)',
        (loc, radius, nsrc, out)
    )

    counter = cur.lastrowid

    # Now add the per-source data
    #
    for n,r,d,inst,obsid,obi,rid,dstk in zip(cr.get_column('name').values,
                                             cr.get_column('ra').values,
                                             cr.get_column('dec').values,
                                             cr.get_column('instrument').values,
                                             cr.get_column('obsid').values,
                                             cr.get_column('obi').values,
                                             cr.get_column('region_id').values,
                                             cr.get_column('detect_stack_id').values):

        db.execute(
            'INSERT INTO sources (src_parent, src_name, src_ra, src_dec, src_instrument, src_obsid, src_obi, src_region_id, src_stack) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (counter, n, float(r), float(d), inst, int(obsid), int(obi), int(rid), dstk)
        )

    db.commit()
    cr = None
    return counter


def read_csc_table(infile, alldata=False):
    """Read in the table and ignore any column which is all NULL

    If alldata is False (default) then only return source-level
    data (ie drop the obsid-level columns) but it does NOT
    remove the repeated rows.
    """

    cr = pycrates.read_file(infile + '[opt kernel=text/tsv]')
    if cr.get_nrows() == 0:
        return cr

    go = []
    skip = False
    for col in cr.get_colnames():
        if not alldata and col == 'obsid':
            skip = True

        if skip:
            go.append(col)
            continue

        values = cr.get_column(col).values
        try:
            if np.isnan(values).all():
                go.append(col)
        except TypeError:
            # assume we can not convert to NaN (e.g. strings)
            pass

    for g in go:
        cr.delete_column(g)

    return cr


def download_src(db, path, name):
    """Download the data files for the source, if needed.

    Parameters
    ----------
    dbase
        The database object.
    path : str
        Location for the file.
    name : str
        The source name.

    """

    outpath = os.path.join(path, name)

    # Have we already downloaded the data?
    rsp = db.execute(
        'SELECT dl_name'
        ' FROM downloaded'
        ' WHERE dl_name = ?',
        (name, )).fetchone()
    if rsp is not None:
        return outpath

    srcs = db.execute(
        'SELECT * FROM sources WHERE src_name = ?',
        (name, )).fetchall()
    if srcs is None:
        return None

    # Marshall into the data we need.
    #
    get = []
    for src in srcs:
        g = {'name': name,
             'obsid': str(src['src_obsid']),
             'obi': str(src['src_obi']),
             'region_id': str(src['src_region_id']),
             'instrument': src['src_instrument'],
             'detect_stack_id': src['src_stack']}
        g['tag'] = f"{src['src_obsid']:05d}_{src['src_obi']:03d}"
        get.append(g)

    files = 'regevt,reg,pha,arf,rmf,lc,psf,regexp,stksrcreg'
    bands = 'broad,wide'
    csccli.retrieve_files(get, path, files, bands, 'all', 'csc2')

    db.execute('INSERT INTO downloaded (dl_name) VALUES (?)',
               (name, ))
    db.commit()
    return outpath


def make_svg(ipath, ftype, infiles):
    """Create or return the SVG for the file."""

    # place the output file into the static area,
    # static/ftype/filename. Note that this is not the
    # default static area, but a location within
    # the instance directory. Since we send the actual
    # contents to the template, and not the file name
    # this is "okay".
    #
    infile = infiles[ftype]
    base = os.path.basename(infile)
    outfile = os.path.join(ftype, base.replace('.fits.gz', '.svg'))
    fullfile = os.path.join(ipath, 'static', outfile)
    if os.path.exists(fullfile):
        return fullfile

    # could only create this directory if we create a file
    outdir = os.path.dirname(fullfile)
    if not os.path.isdir(outdir):
        # NOTE: should probably check permission
        os.mkdir(outdir)

    if ftype == 'lc3':
        return make_svg_lc3(infile, fullfile)

    if ftype == 'pha3':
        return make_svg_pha3(infile, fullfile)

    if ftype == 'regexp3':
        return make_svg_regexp3(infile, infiles['reg3'],
                                infiles['stksrcreg'],
                                fullfile)

    if ftype == 'regevt3':
        return make_svg_regevt3(infile, infiles['reg3'],
                                infiles['stksrcreg'],
                                fullfile)

    if ftype == 'psf3':
        return make_svg_psf3(infile, infiles['reg3'], fullfile)

    return None


def make_svg_lc3(infile, outfile):
    """Create LC3 SVG plot.

    Should add info from the header, but what's interesting?
    """

    cr = pycrates.read_file(infile)

    # How to convert Mission-Elapsed-Time to a useful value?
    tstart = cr.get_key('TSTART').value

    tstart_utc = dates(tstart).strftime('%Y-%m-%d %H:%M:%S')

    band = cr.get_key('BAND').value
    ap = cr.get_key('APERTURE').value

    obsid = int(cr.get_key('OBS_ID').value)
    obi = int(cr.get_key('OBI_NUM').value)
    det = f'{obsid:05d}_{obi:03d}'

    t = cr.get_column('Time').values
    r = cr.get_column('COUNT_RATE').values
    dr = cr.get_column('COUNT_RATE_ERR').values

    lo = cr.get_column('Minus_3Sig').values
    hi = cr.get_column('Plus_3Sig').values

    dt = t - tstart

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.fill_between(dt, lo, hi, alpha=0.2, label=r'3$\sigma$ range')
    ax.errorbar(t - tstart, r, yerr=dr, fmt='o', c='k')

    # add in the background signal, approximately scaled
    #
    bcr = pycrates.read_file(infile + '[BKGLIGHTCURVE]')
    bap = cr.get_key('APERTURE').value

    scale = ap / bap

    t = bcr.get_column('Time').values
    # r = cr.get_column('COUNT_RATE').values
    # dr = cr.get_column('COUNT_RATE_ERR').values

    lo = scale * bcr.get_column('Minus_3Sig').values
    hi = scale * bcr.get_column('Plus_3Sig').values

    dt = t - tstart
    ax.fill_between(dt, lo, hi, alpha=0.2, label=r'3$\sigma$ range (background)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (count/s)')
    ax.set_title(f'{det} - {band}')

    ax.legend()

    # add an indication of TSTART
    #
    ax.text(0.05, 0.95, f'Start: {tstart_utc}',
            transform=fig.transFigure)

    fig.savefig(outfile)
    plt.close(fig)
    return outfile


def make_svg_pha3(infile, outfile):
    """Create pha3 SVG plot.

    Should add info from the header, but what's interesting?

    Really would want this to be interactive.
    """

    d = io.read_pha(infile)
    d.name = os.path.basename(infile)

    d.notice(0.5, 7)
    d.group_counts(15, tabStops=~d.mask)
    d.subtract()

    # Want to use Chi2DataVar but can not guarantee the > 0
    # requirements.
    #
    plot = DataPHAPlot()
    plot.prepare(d, stat=Chi2Gehrels())

    fig = plt.figure(figsize=(8, 6))
    plot.plot()

    fig.savefig(outfile)
    plt.close(fig)
    return outfile


def add_ellipses(cr, color='w', linestyle='-'):
    """Add ellipses from a region crate.

    """

    thetas = np.linspace(0, np.pi * 2, 50)

    for i in range(cr.get_nrows()):
        shape = cr.get_column('shape').values[i]

        if shape not in ['Ellipse', '!Ellipse']:
            print(f'Non ellipse region shape: {shape}')
            continue

        x0 = cr.get_column('x').values[i]
        y0 = cr.get_column('y').values[i]
        r = cr.get_column('r').values[i]
        rmaj = r[0]
        rmin = r[1]
        theta = cr.get_column('rotang').values[i][0] * np.pi / 180
        if rmaj < rmin:
            print(f'rmajor/minor problem: {rmaj} {rmin}')
            continue

        # Create an ellipse using polar coordinates
        # https://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center
        #
        thetas2 = thetas - theta
        rbot2 = (rmin * np.cos(thetas2))**2 + (rmaj * np.sin(thetas2))**2
        r = rmaj * rmin / np.sqrt(rbot2)

        x = x0 + r * np.cos(thetas)
        y = y0 + r * np.sin(thetas)

        col = 'red' if shape == '!Ellipse' else color
        plt.plot(x, y, c=col, ls=linestyle)


def add_psf_regions(regfile, color='w'):
    """Add on the PSF region to the image."""

    # going to assume the first block is what we want
    # (not ideal)
    #
    rcr = pycrates.read_file(regfile)
    add_ellipses(rcr, color=color)


def add_stack_regions(regfile, color='w'):
    """Add on the 'stack' region to the image."""

    ds = pycrates.CrateDataset(regfile, mode='r')

    src = ds.get_crate('SRCREG')
    add_ellipses(src, color='cyan')

    src = ds.get_crate('BKGREG')
    add_ellipses(src, color='cyan', linestyle='--')


def make_svg_regexp3(infile, regfile, stkregfile, outfile):
    """Create regexp3 SVG plot.

    Overlays the PSF shape, and normalizes by the
    median value (as a percentage).
    """

    icr = pycrates.read_file(infile)
    band = icr.get_key('BAND').value

    # convert to a %fraction compared to the median value
    # (since the absolute values here aren't that useful)
    #
    idata = icr.get_image().values
    med = np.median(idata)
    idata = 100 * (idata - med) / med

    # what's the SKY range; assume only a single component
    xsub = icr.get_subspace_data(1, 'x')
    ysub = icr.get_subspace_data(1, 'y')

    xmin = xsub.range_min[0]
    xmax = xsub.range_max[0]
    ymin = ysub.range_min[0]
    ymax = ysub.range_max[0]
    extent = [xmin, xmax, ymin, ymax]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(idata, cmap='viridis', origin='lower',
                   aspect='equal', extent=extent)

    # ensure the limits don't change when we add in regions
    #
    ax.axis('image')

    cbar = fig.colorbar(mappable=im, ax=ax)
    cbar.set_label('% change from median value')

    plt.title(f'Exposure map: {band}')

    add_psf_regions(regfile)
    add_stack_regions(stkregfile)

    fig.savefig(outfile)
    plt.close(fig)
    return outfile


def make_svg_psf3(infile, regfile, outfile):
    """Create psf3 SVG plot.

    For now show the PSF and not BINPSF block.
    """

    icr = pycrates.read_file(infile)
    band = icr.get_key('BAND').value

    idata = icr.get_image().values

    # what's the SKY range; assume only a single component
    xsub = icr.get_subspace_data(1, 'x')
    ysub = icr.get_subspace_data(1, 'y')

    xmin = xsub.range_min[0]
    xmax = xsub.range_max[0]
    ymin = ysub.range_min[0]
    ymax = ysub.range_max[0]
    extent = [xmin, xmax, ymin, ymax]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    lnorm = LogNorm(vmin=1, vmax=idata.max())
    im = ax.imshow(idata, cmap='viridis', norm=lnorm,
                   origin='lower',
                   aspect='equal', extent=extent)
    cbar = fig.colorbar(mappable=im, ax=ax)
    cbar.set_label('PSF counts')

    plt.title(f'PSF map: {band}')

    add_psf_regions(regfile)

    fig.savefig(outfile)
    plt.close(fig)
    return outfile


def make_svg_regevt3(infile, regfile, stkregfile, outfile):
    """Create regevt3 SVG plot.

    Overlays the PSF shape.
    """

    ds = pycrates.CrateDataset(infile, mode='r')
    inst = ds.get_crate('PRIMARY').get_key('INSTRUME').value

    # Assume there's only one mask block
    mcr = ds.get_crate('MASK')
    mdata = mcr.get_image().values.copy()  # do we want this?
    xsub = mcr.get_subspace_data(1, 'x')
    ysub = mcr.get_subspace_data(1, 'y')
    xmin = xsub.range_min[0]
    xmax = xsub.range_max[0]
    ymin = ysub.range_min[0]
    ymax = ysub.range_max[0]
    extent = [xmin, xmax, ymin, ymax]

    mcr = None
    ds = None

    # bin into an image
    if inst == 'ACIS':
        icr = pycrates.read_file(f"{infile}[energy=500:7000][bin x={xmin}:{xmax},y={ymin}:{ymax}]")
        band = 'broad'
    elif inst == 'HRC':
        raise IOError("what binning/cols do we want?")
        icr = pycrates.read_file(f"{infile}[bin x={xmin}:{xmax},y={ymin}:{ymax}]")
        band = 'wide'
    else:
        raise IOError(f"Invalid INSTRUME={inst}")

    idata = icr.get_image().values

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(idata, cmap='viridis', origin='lower',
                   aspect='equal', extent=extent)
    cbar = fig.colorbar(mappable=im, ax=ax)
    cbar.set_label('Counts')

    plt.title(f'Source counts: {band}')

    add_psf_regions(regfile)
    add_stack_regions(stkregfile)

    fig.savefig(outfile)
    plt.close(fig)
    return outfile


def read_svg(infile):
    """Return the svg tag contents.

    This strips out the XML declaration/doctype.
    """

    cts = open(infile, 'r').read()
    idx = cts.find('<svg')
    if idx == -1:
        return "Unable to read input file"

    return cts[idx:]
