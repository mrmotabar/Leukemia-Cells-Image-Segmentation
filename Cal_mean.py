start = int(input())
end = int(input())
sum_cost = 0
sum_cost_255 = 0
sum_psnr = 0
for file_number in range(start, end + 1):
    f = open(".\\Results_5\\Im" + "{:03d}".format(file_number) + "_" + str(1 - file_number // 131) + ".txt", "r")
    lines = f.readlines()
    cost = float(lines[0][6:])
    cost_255 = float(lines[1][10:])
    psnr = sum(list(map(float, lines[2][8: -2].split()))) / 3
    sum_cost += cost
    sum_cost_255 += cost_255
    sum_psnr += psnr
    f.close()
print("Cost_mean")
print(sum_cost / (end - start + 1))
print("Cost_255_mean")
print(sum_cost_255 / (end - start + 1))
print("PSNR_mean")
print(sum_psnr / (end - start + 1))