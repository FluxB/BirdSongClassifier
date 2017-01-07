import sys
import queue

fg = open(sys.argv[1], 'r')
prominent = open(sys.argv[2], 'w')
number_prominent = int(sys.argv[3])

label_dict = {}
for line in fg:
    path, label = line.split(' ')
    label = label.strip()
    label_dict.setdefault(label, []).append(path)

prio_queue = queue.PriorityQueue()
for label, paths in label_dict.items():
    prio_queue.put((-len(paths), label))

i = 0
while not prio_queue.empty():
    (count, label) = prio_queue.get()
    print(label, count)
    paths = label_dict[label]
    for path in paths:
        prominent.write(str.format("{} {}\n", path, str(i)))

    i += 1
    if i == number_prominent:
        break
