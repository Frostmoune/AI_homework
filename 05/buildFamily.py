with open('testFamily.pl', 'w') as f:
    max_depth = 5
    f.write("male(%s).\n"%('s'))
    def createFamily(name, depth):
        if depth > max_depth:
            return
        f.write("child(%s, %s).\n"%(name + 'a', name))
        f.write("male(%s).\n"%(name + 'a'))
        createFamily(name + 'a', depth + 1)
        f.write("child(%s, %s).\n"%(name + 'b', name))
        f.write("male(%s).\n"%(name + 'b'))
        createFamily(name + 'b', depth + 1)
    createFamily("s", 0)