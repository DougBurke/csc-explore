# Can we explore the CSC 2.0 "nicely"?
#

import os

import numpy as np

from flask import Flask, flash, redirect, render_template, request, url_for

import pycrates

from ciao_contrib.cda import csccli
from ciao_contrib.runtool import make_tool

from . import dbase


def create_app():

    # Currently there's no way to over-ride the settings.
    #
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'explore.sqlite'),
    )

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
        return f'Something about {name}'

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
             'instrument': src['src_instrument']}
        g['tag'] = f"{src['src_obsid']:05d}_{src['src_obi']:03d}"
        print(g['tag'])
        get.append(g)

    files = 'regevt,pha,arf,rmf,lc,psf,regexp'
    bands = 'broad,wide'
    csccli.retrieve_files(get, path, files, bands, 'all', 'csc2')

    db.execute('INSERT INTO downloaded (dl_name) VALUES (?)',
               (name, ))
    db.commit()
    return outpath
