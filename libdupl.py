def dupl(diro,labelsl):
    import os
    import sys
    sys.path.append(os.path.abspath("/home/et/atools/"))
    import libfreq
    import libmaxa
    sl,nl,fre = libfreq.freq(labelsl[:],srt=False)
    mnl = max(nl)
    repnl = [round(mnl/x) for x in nl] #repition list.. This is very similar to fre in libfreq.freq
    directof = [str()]*sum([repnl[i]*nl[i] for i in range(len(nl))])#(max(nl)*len(sl))
    ii = 0
    for i,j in enumerate(diro):
        #print(directof,ii)
        rnldum = repnl[sl.index(labelsl[i])]
        directof[ii:ii+rnldum] = [j]*rnldum
        ii+= rnldum
    
    #maxlb2,labelsl2 = libmaxa.maxa(directof[:])
    #sl2,nl2,fre2 = libfreq.freq(labelsl2)
    
    #b = list(range(a))
    #c = list(zip(a,b))
    #c.sort()
    #d = [x[1] for x in c] #returns index list
    return directof
#import os
#os.chdir("/home/et/pytt/geo/adata_100k_20w/data")
#dirf = os.listdir()
#a = dupl(dirf,"/home/et/adata_100k_20w/data/")
