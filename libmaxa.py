def maxa(a,n=4):
    for i,j in enumerate(a):
        try:
            a[i] = float(j.split("_")[1][:-n])
        except:
            a.pop(i)
            a[i] = float(a[i].split("_")[1][:-n])
    return max(a),a
