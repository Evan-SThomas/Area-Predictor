def freq(a,srt=True):
    if srt==True:
        a.sort()
    sl = list(set(a))
    nl = [a.count(x) for x in sl]
    mnl = max(nl)
    f = 'Nan'
#    f = [x/mnl for x in nl]
    return sl,nl,f
