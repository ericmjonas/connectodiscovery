import pandas
import sqlite3

dbname = "../../../preprocess/mouseretina/mouseretina.db"

con = sqlite3.connect(dbname)
MAX_CONTACT_AREA=5.0
area_thold_min = 0.1

contacts_df = pandas.io.sql.read_frame("select * from contacts where area < %f and area > %f" % (MAX_CONTACT_AREA, area_thold_min), 
                                       con, index_col='id')

 contacts_df.head()

