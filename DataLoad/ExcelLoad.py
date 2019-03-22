import xlrd
from pprint import pprint

class ExcelLoad:

    def LoadData(filepath,sheetName="Sheet1"):
        wb=xlrd.open_workbook(filepath)
        ws=wb.sheet_by_name(sheetName)
        dataset=[]
        for r in range(ws.nrows):
            col=[]
            for c in range(ws.ncols):
                col.append(ws.cell(r,c).value)
            dataset.append(col)
        pprint(dataset)
        return dataset

filename=r"E:\Folder Code\MachineLearning\test.xlsx"
data=ExcelLoad.LoadData(filename)
print(len(data))

