import csv
import sys

class CsvLoad:

    @classmethod
    def LoadData(filepath,delimiter=','):
        try:
            with open(filepath) as f:
                reader=csv.reader(f)
            header=reader.next()
            data=[row for row in header]
            return header,data
        except csv.Error as e:
            print(e)
            sys.exit()