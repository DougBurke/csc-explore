DROP TABLE IF EXISTS searches;
DROP TABLE IF EXISTS sources;
DROP TABLE IF EXISTS sources2;
DROP TABLE IF EXISTS downloaded;

CREATE TABLE searches (
  counter INTEGER PRIMARY KEY AUTOINCREMENT,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  location TEXT NOT NULL,
  radius REAL NOT NULL,
  nsrc INTEGER NOT NULL,
  filename TEXT UNIQUE NOT NULL
);

CREATE TABLE sources (
  src_parent INTEGER NOT NULL,
  src_name TEXT NOT NULL,
  src_ra REAL NOT NULL,
  src_dec REAL NOT NULL,
  src_instrument TEXT NOT NULL,
  src_obsid INTEGER NOT NULL,
  src_obi INTEGER NOT NULL,
  src_region_id INTEGER NOT NULL,
  src_stack TEXT NOT NULL,
  FOREIGN KEY (src_parent) REFERENCES searches (counter)
);

CREATE TABLE downloaded (
  dl_name TEXT UNIQUE NOT NULL
);
