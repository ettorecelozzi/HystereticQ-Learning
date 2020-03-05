import numpy as np


def countNot0(qTables):
    """
    Count the number of element that are not 0 in the QTables
    :param qTables: list of the qTables
    :return: number of elment != 0
    """
    counter = [0] * len(qTables)
    qTable_index = 0
    for q in qTables:
        for state in q:
            for action in q[state]:
                if q[state][action] != 0.0:
                    counter[qTable_index] += 1
        qTable_index += 1
    print(counter)


def printProgressBar(iteration, total, prefix='Progress:', suffix='', decimals=1, length=45, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    Print fancy progress bars. From Greenstick at StackOverflow
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    usage:
    for i in range(workamount):
        doWork(i)
        printProgressBar(i, workamount)

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
