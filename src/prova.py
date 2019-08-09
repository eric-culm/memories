



a = {'a':1,'b':2,'c':3}

b = {'u':1,'d':2,'c':3}

for i in a.keys():
    if i in b.keys():
        del b[i]

print (b)
