from flask.cli import FlaskGroup
from flask_cors import CORS

import psycopg2
from project import app, db

CORS(app)

cli = FlaskGroup(app)


@cli.command("create_db")
def create_db():
    db.drop_all()
    db.create_all()
    db.session.commit()


if __name__ == "__main__":
    cli()
