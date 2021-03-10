"""
Make-up-your-database-as-you-go-along as-a-service

"""

import glob
import os
import sqlite3

import click
from flask import current_app, g
from flask.cli import with_appcontext


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db():
    """Not only do we delete the DB but we clean out the
    instance directory."""

    path = current_app.instance_path
    print(f"Cleaning up: {path}")

    # Only delete the files in this directory, not any
    # sub-directories (for now at least).
    #
    pat = os.path.join(path, '*')
    for match in glob.glob(pat):
        # should we not remove the database?
        if os.path.isfile(match):
            os.remove(match)

    # Create the static area. Should probably clean it out
    # if it already exists.
    #
    static = os.path.join(path, 'static')
    if os.path.isdir(static):
        # laxy way to delete the directory
        # os.rmdir(static)
        os.system(f"rm -rf {static}")

    os.mkdir(static)

    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))




@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
