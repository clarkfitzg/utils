#!/usr/bin/env python

'''
Runs a Python script and generates a report
'''

import sys
import io


pyfile = sys.argv[1]

outfile = pyfile.replace('.py', '.txt')

begin = '''
------------------------------------------------------------
BEGIN {pyfile}
------------------------------------------------------------

'''.format(pyfile=pyfile)

end = '''
------------------------------------------------------------
END {pyfile}
------------------------------------------------------------

------------------------------------------------------------
OUTPUT
------------------------------------------------------------

'''.format(pyfile=pyfile)


with open(outfile, 'w') as out, open(pyfile) as f:

    # File contents
    out.write(begin)
    for line in f:
        out.write(line)
    out.write(end)

    # Capture stdout with a buffer
    buff = io.StringIO()
    sys.stdout = buff

    # Output after code is run
    f.seek(0)
    exec(f.read())
    print(buff.getvalue(), file=out)

    # Restore stdout
    sys.stdout = sys.__stdout__
