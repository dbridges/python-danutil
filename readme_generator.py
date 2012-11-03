import sys
import danutil

if len(sys.argv) != 2:
    print 'Usage python mark_doc_gen.py INPUTFILE'
    sys.exit(-1)

filename = sys.argv[1]

functions = []

startstring = '# python-danutil\nA collection of useful functions and classes' +\
              ' for use in data analysis.\n'
outstring = ''

with open(filename.lower()) as f:
        for line in f:
            if 'def ' in line and 'self' not in line and 'def _' not in line:
                functions.append(line.strip()[4:-1])
            elif 'class' in line:
                functions.append(line.strip()[6:-1])
            elif '### ' in line:
                functions.append(line.strip())

toc_string = '## Documentation \n python-danutil provides functions and classes ' +\
             'in these general areas:\n'

for function in functions:
    if function.startswith('#'):
        outstring += '\n' + function + '\n'
        linkname = function[4:].lower().replace(' ', '-')
        toc_string += '* [' + function[4:] + '](#' + linkname + ')\n'
        continue
    func_name = function.split('(')[0]
    outstring += '```\n' + function + '\n'
    outstring += eval(filename[:-3] + '.' + func_name).__doc__ + '\n```\n\n'

outstring = startstring + toc_string + outstring

with open('README.md', 'w') as f:
    f.write(outstring)
