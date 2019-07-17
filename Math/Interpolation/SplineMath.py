

def SplineInsertPoint(xlist, ylist, xs, chf):
    if(len(xlist) != len(ylist)):
        return False
    length=len(xlist)
    h,f,l,v,g=[],[],[],[],[]

    for i in range(length-1):
        hthis=xlist[i+1]-xlist[i]
        h.append(hthis)
        g.append((ylist[i+1]-ylist)/hthis

    for i in range(length-1):
        