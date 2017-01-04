import sys

fg_labels = open(sys.argv[1], 'r')
bg_labels = open(sys.argv[2], 'r')
fg_bg_merged = open(sys.argv[3], 'w')

number_fg = 0
for fg in fg_labels:
    fg_path = fg.split(' ')[0]
    fg_bg_merged.write(str.format("{} 0\n", fg_path))
    number_fg += 1

print("Number of foreground files", number_fg)

number_bg = 0
for bg in bg_labels:
    bg_path = bg.split(' ')[0]
    fg_bg_merged.write(str.format("{} 1\n", bg_path))
    number_bg += 1

print("Number of background files", number_bg)
