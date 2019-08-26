def score(lf,netf,data,max_area):
    import h5py
    import os
    import sys
    sys.path.append(os.path.abspath("/home/et/pytt/geo/atools/"))
    import libfreq
    import libmaxa
    import libarea
    import torch
    import numpy as np
    import torch
#    dum,al = libmaxa.maxa(lf[:])
#    sl,nl,fr= libfreq.freq(al)
    f = [data[x][:] for x in lf] #200
    lbl = lf[:]
    al = [data[x].attrs['area']/max_area for x in lbl]
    sl = list(set(al))
    nl = [al.count(x) for x in sl]
    #testing
    #lbl = [data['t'][x].attrs['area']/10.0 for x in lf]
    netl = [float(netf(torch.from_numpy(np.float32(data[x][:])).reshape((1,1,20,20)))) for x in lf]
#    netl = [float(netf.forward(x)[0][0])*10 for x in f]
    dif = [abs(netl[i]-al[i]) for i in range(len(lbl))]
    #sl = list(set(lbl))
    score = torch.zeros(len(sl))
    average_prediction = torch.zeros(len(sl))
    for i in range(len(lbl)):
        score[int(sl.index(al[i]))] += dif[i]
        average_prediction[int(sl.index(al[i]))] += netl[i]
    average_prediction = [float(average_prediction[i]/nl[i]) for i in range(len(nl))]
    normed_score = [float(score[i]/nl[i]) for i in range(len(nl))] #Normed score is average diff
    try: 
        zero_index = al.index(0.0)
        nszi = normed_score[:]
        slzi = sl[:]
        nszi.pop(zero_index)
        slzi.pop(zero_index)
        rel_err = [float(nszi[i]/slzi[i]) for i in range(len(slzi))]
    except:
        rel_err = [float(normed_score[i]/sl[i]) for i in range(len(sl))]
    return sl, nl, normed_score, average_prediction, rel_err

