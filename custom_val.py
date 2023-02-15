import os

base_path = "custom_data/train/labels"
labels = os.listdir(base_path)
print(len(labels))
class_0 = 0
class_1 = 0
class_2 = 0
class_3 = 0

for i in labels:
    f = open(base_path + "/" + i)
    for j in f:
        if int(j[0])==0: class_0 += 1
        elif int(j[0])==1: class_1 += 1
        elif int(j[0])==2: class_2 += 1
        elif int(j[0])==3: class_3 += 1

print("class 0:", class_0)
print("class 1:", class_1)
print("class 2:", class_2)
print("class 3:", class_3)

# base_path = "custom_data/valid/labels"
# labels = os.listdir(base_path)

# for i in labels:
#     f = open(base_path+"/"+i, "r")
#     content = str("")
#     for j in f:
#         if int(j[0])==1: content = content + j
#         elif int(j[0])==3:
#             content = content + "0" + j[1:]
        
#     content = content.rstrip("\n")
    
#     g = open(base_path+"/"+i, "w")
#     g.write(content)


# a = "1koskd\nsodkosdks\n"
# b = a.strip("\n")
# print(a)