from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torch
import clip
import json

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
model, preprocess = clip.load("ViT-B/32", device=device)



class SaliencyDatasetWithText(Dataset):
    def __init__(self, image_paths, target_paths, text_sequences, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.text_sequences = []
        for i in range(len(image_paths)):
            # 处理文本
            text_tokens = clip.tokenize([text_sequences[i]]).to(device)

            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                self.text_sequences.append(text_features)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        target = Image.open(self.target_paths[idx]).convert('L')
        text = self.text_sequences[idx]

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        # 将文本序列转换为 PyTorch 张量
        text_tensor = torch.tensor(text, dtype=torch.long)

        return image, target, text_tensor


    
# 单纯划分数据用
def split_dataset(image_paths, target_paths, text_descriptions, train_ratio=0.7, val_ratio=0.15):
    combined = list(zip(image_paths, target_paths, text_descriptions))
    random.shuffle(combined)

    total_images = len(combined)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)

    train_data = combined[:train_size]
    val_data = combined[train_size:train_size + val_size]
    test_data = combined[train_size + val_size:]
    # print(type(test_data))
    # print(len(test_data))
    # print(test_data[2])
    # list1 = [330, 218, 138, 206, 326, 512, 58, 84, 96, 180, 592, 422, 300, 86, 142, 528, 496, 74, 6, 554, 176, 384, 304, 236, 484, 310, 460, 266, 362, 354, 318, 458, 196, 252, 78, 504, 342, 174, 350, 60, 250, 352, 506, 190, 366, 292, 332, 256, 448, 38, 346, 12, 152, 70, 114, 424, 0, 290, 232, 546, 126, 186, 502, 514, 100, 140, 414, 364, 188, 172, 548, 154, 532, 132, 334, 538, 2, 76, 408, 234, 390, 568, 286, 194, 578, 146, 530, 382, 446, 88, 260, 520, 344, 258, 534, 246, 500, 470, 432, 466, 420, 80, 102, 444, 16, 148, 92, 542, 418, 434, 162, 192, 598, 90, 94, 284, 574, 452, 130, 536, 278, 144, 10, 50, 34, 492, 586, 296, 372, 178, 136, 494, 550, 238, 204, 440, 570, 282, 348, 402, 404, 42, 450, 396, 116, 294, 112, 104, 340, 68, 208, 170, 522, 516, 198, 524, 320, 36, 368, 224, 24, 526, 328, 220, 182, 576, 122, 322, 482, 376, 276, 590, 226, 274, 8, 72, 150, 118, 18, 164, 52, 306, 160, 596, 166, 474, 20, 562, 32, 386, 438, 398, 442, 202, 268, 476, 400, 272, 28, 566, 374, 370, 558, 436, 338, 30, 464, 410, 14, 168]
    # list2 = [4, 508, 216, 472, 106, 270, 486, 212, 312, 98, 426, 462, 222, 254, 302, 406, 54, 588, 488, 48, 184, 64, 324, 454, 456, 394, 66, 336, 360, 556, 314, 200, 358, 490, 230, 378, 158, 594, 298, 416, 82, 552, 478, 108, 26]
    # list3 = [128, 388, 518, 134, 262, 264, 392, 22, 280, 540, 156, 412, 544, 288, 40, 44, 428, 46, 430, 560, 564, 308, 56, 572, 316, 62, 580, 582, 584, 210, 468, 214, 380, 248, 480, 228, 356, 110, 240, 242, 498, 244, 120, 124, 510]
    # list1 = [746, 134, 840, 908, 740, 748, 758, 354, 76, 110, 528, 4, 658, 1096, 616, 202, 126, 184, 252, 186, 956, 670, 604, 1154, 412, 94, 402, 864, 682, 576, 540, 234, 490, 464, 344, 686, 954, 874, 610, 756, 294, 1034, 226, 726, 152, 56, 358, 144, 690, 770, 1046, 844, 436, 162, 724, 1138, 836, 372, 86, 16, 1092, 54, 888, 338, 1064, 20, 872, 512, 44, 492, 734, 620, 1152, 732, 324, 260, 594, 12, 1070, 992, 564, 1120, 136, 102, 708, 22, 238, 196, 1012, 626, 936, 422, 1112, 1162, 206, 64, 1004, 342, 764, 1060, 870, 1160, 1098, 1110, 766, 558, 810, 386, 774, 596, 568, 952, 1100, 288, 390, 258, 316, 640, 414, 928, 406, 970, 584, 996, 280, 70, 950, 200, 1024, 1182, 440, 1176, 1158, 646, 960, 1148, 982, 498, 714, 1058, 494, 246, 376, 326, 818, 562, 42, 972, 530, 446, 450, 1036, 906, 172, 778, 750, 366, 80, 1054, 780, 1142, 330, 882, 164, 668, 792, 718, 108, 904, 572, 180, 574, 892, 966, 762, 520, 148, 198, 790, 1094, 580, 128, 310, 694, 548, 910, 552, 104, 598, 524, 702, 72, 236, 204, 2, 824, 712, 84, 384, 48, 794, 656, 266, 1116, 380, 754, 274, 32, 1146, 964, 350, 1156, 862, 140, 378, 52, 614, 496, 986, 722, 590, 842, 696, 876, 838, 1026, 472, 728, 100, 612, 1170, 438, 484, 990, 632, 588, 976, 106, 216, 1188, 648, 312, 802, 182, 886, 272, 926, 642, 1086, 916, 332, 8, 710, 208, 10, 890, 1050, 38, 518, 930, 166, 46, 130, 90, 984, 224, 542, 296, 786, 248, 6, 866, 292, 846, 488, 482, 478, 998, 1078, 674, 1014, 214, 608, 96, 704, 98, 360, 396, 1136, 716, 368, 902, 1194, 232, 328, 592, 474, 1164, 174, 1168, 532, 1028, 284, 220, 832, 1006, 638, 666, 566, 352, 662, 814, 210, 676, 156, 58, 538, 250, 962, 26, 270, 544, 1056, 300, 454, 468, 988, 1082, 276, 176, 1102, 1166, 856, 36, 812, 586, 416, 826, 116, 240, 942, 334, 932, 408, 340, 678, 112, 382, 1174, 346, 1066, 602, 452, 1088, 1032, 132, 898, 804, 466, 788, 442, 820, 362, 74, 1106, 424, 848, 476, 218, 364, 800, 546, 880, 92, 934, 700, 854, 1044, 286, 78, 28, 1114, 426, 940, 444, 1022, 500, 978, 744, 994, 1030, 1126, 1186, 536, 138, 462, 1190, 1198, 688, 1076, 1180, 860, 308, 834, 650, 600, 192, 150, 428, 1196, 706, 1090, 1048, 944, 664, 922, 672, 160, 1128, 760]
    # list2 = [1104, 142, 222, 720, 120, 178, 1062, 1132, 40, 370, 550, 730, 652, 920, 776, 456, 320, 784, 506, 264, 1150, 118, 578, 34, 432, 534, 554, 480, 1068, 782, 510, 154, 624, 948, 302, 634, 894, 1184, 1172, 742, 644, 158, 434, 806, 738, 1040, 1144, 692, 526, 194, 62, 684, 556, 1008, 1074, 772, 170, 768, 918, 974, 60, 82, 1072, 628, 356, 560, 0, 828, 486, 502, 314, 796, 188, 606, 244, 256, 1140, 736, 852, 830, 212, 1122, 1178, 798, 946, 304, 858, 146, 228, 254]
    # list3 = [514, 516, 522, 14, 1038, 18, 1042, 24, 1052, 30, 50, 1080, 570, 1084, 66, 68, 582, 1108, 88, 1118, 1124, 618, 1130, 622, 1134, 114, 630, 122, 636, 124, 654, 660, 168, 680, 1192, 698, 190, 230, 752, 242, 262, 268, 278, 282, 290, 808, 298, 816, 306, 822, 318, 322, 336, 850, 348, 868, 878, 884, 374, 896, 388, 900, 392, 394, 398, 400, 912, 914, 404, 410, 924, 418, 420, 938, 430, 958, 448, 968, 458, 460, 980, 470, 1000, 504, 1002, 508, 1010, 1016, 1018, 1020]
    # list1 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 60, 61, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 78, 79, 80, 81, 82, 83, 84, 85, 88, 89, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 118, 119, 120, 121, 122, 123, 124, 125, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 154, 155, 156, 157, 158, 159, 160, 161, 164, 165, 166, 167, 170, 171, 172, 173, 174, 175, 176, 177, 184, 185, 186, 187, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 210, 211, 212, 213, 216, 217, 218, 219, 220, 221, 222, 223, 228, 229, 230, 231, 232, 233, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 334, 335, 336, 337, 340, 341, 342, 343, 344, 345, 346, 347, 352, 353, 360, 361, 364, 365, 366, 367, 368, 369, 372, 373, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 394, 395, 396, 397, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 490, 491, 492, 493, 494, 495, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 526, 527, 530, 531, 536, 537, 538, 539, 540, 541, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 572, 573, 574, 575, 576, 577, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 616, 617, 618, 619, 624, 625, 626, 627, 642, 643, 644, 645, 646, 647, 650, 651, 652, 653, 654, 655, 656, 657, 664, 665, 666, 667, 668, 669, 670, 671, 674, 675, 680, 681, 682, 683, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 704, 705, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 776, 777, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 804, 805, 806, 807, 808, 809, 812, 813, 814, 815, 816, 817, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 900, 901, 902, 903, 914, 915, 918, 919, 920, 921, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 942, 943, 944, 945, 946, 947, 950, 951, 952, 953, 958, 959, 960, 961, 962, 963, 964, 965, 968, 969, 970, 971, 972, 973, 986, 987, 988, 989, 990, 991, 992, 993, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1036, 1037, 1038, 1039, 1040, 1041, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1070, 1071, 1072, 1073, 1078, 1079, 1084, 1085, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1132, 1133, 1134, 1135, 1140, 1141, 1144, 1145, 1146, 1147, 1150, 1151, 1152, 1153, 1154, 1155, 1160, 1161, 1164, 1165, 1170, 1171, 1172, 1173, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1228, 1229, 1240, 1241, 1242, 1243, 1244, 1245, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1276, 1277, 1278, 1279, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1322, 1323, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1462, 1463, 1464, 1465, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1574, 1575, 1576, 1577, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1606, 1607, 1608, 1609, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1682, 1683, 1684, 1685, 1688, 1689, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1722, 1723, 1724, 1725, 1726, 1727, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799]
    # list2 = [0, 1, 16, 17, 56, 57, 58, 59, 86, 87, 90, 91, 114, 115, 126, 127, 150, 151, 162, 163, 168, 169, 180, 181, 182, 183, 214, 215, 234, 235, 236, 237, 258, 259, 282, 283, 284, 285, 356, 357, 358, 359, 374, 375, 398, 399, 400, 401, 488, 489, 522, 523, 524, 525, 532, 533, 534, 535, 556, 557, 568, 569, 570, 571, 600, 601, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 658, 659, 660, 661, 662, 663, 672, 673, 684, 685, 686, 687, 688, 689, 706, 707, 738, 739, 740, 741, 768, 769, 770, 771, 772, 773, 774, 775, 800, 801, 802, 803, 810, 811, 840, 841, 904, 905, 906, 907, 908, 909, 922, 923, 924, 925, 940, 941, 948, 949, 954, 955, 956, 957, 978, 979, 980, 981, 1034, 1035, 1042, 1043, 1068, 1069, 1074, 1075, 1076, 1077, 1080, 1081, 1082, 1083, 1100, 1101, 1124, 1125, 1126, 1127, 1128, 1129, 1136, 1137, 1138, 1139, 1142, 1143, 1148, 1149, 1156, 1157, 1158, 1159, 1162, 1163, 1166, 1167, 1168, 1169, 1176, 1177, 1196, 1197, 1198, 1199, 1200, 1201, 1220, 1221, 1222, 1223, 1246, 1247, 1248, 1249, 1268, 1269, 1270, 1271, 1272, 1273, 1280, 1281, 1282, 1283, 1324, 1325, 1326, 1327, 1360, 1361, 1362, 1363, 1380, 1381, 1382, 1383, 1402, 1403, 1404, 1405, 1466, 1467, 1468, 1469, 1494, 1495, 1496, 1497, 1506, 1507, 1508, 1509, 1570, 1571, 1572, 1573, 1602, 1603, 1604, 1605, 1610, 1611, 1612, 1613, 1638, 1639, 1640, 1641, 1678, 1679, 1680, 1681, 1686, 1687, 1690, 1691, 1738, 1739, 1748, 1749, 1750, 1751, 1764, 1765, 1766, 1767, 1776, 1777, 1778, 1779]
    # list3 = [20, 21, 22, 23, 30, 31, 54, 55, 62, 63, 70, 71, 76, 77, 116, 117, 152, 153, 178, 179, 188, 189, 190, 191, 204, 205, 206, 207, 208, 209, 224, 225, 226, 227, 256, 257, 286, 287, 302, 303, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 338, 339, 348, 349, 350, 351, 354, 355, 362, 363, 370, 371, 390, 391, 392, 393, 418, 419, 420, 421, 436, 437, 438, 439, 460, 461, 462, 463, 464, 465, 466, 467, 486, 487, 496, 497, 498, 499, 520, 521, 528, 529, 542, 543, 578, 579, 580, 581, 614, 615, 620, 621, 622, 623, 648, 649, 676, 677, 678, 679, 702, 703, 742, 743, 744, 745, 778, 779, 780, 781, 818, 819, 856, 857, 878, 879, 880, 881, 896, 897, 898, 899, 910, 911, 912, 913, 916, 917, 966, 967, 974, 975, 976, 977, 982, 983, 984, 985, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1052, 1053, 1054, 1055, 1064, 1065, 1066, 1067, 1086, 1087, 1088, 1089, 1130, 1131, 1174, 1175, 1224, 1225, 1226, 1227, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1274, 1275, 1320, 1321, 1340, 1341, 1342, 1343, 1364, 1365, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1434, 1435, 1436, 1437, 1458, 1459, 1460, 1461, 1478, 1479, 1480, 1481, 1538, 1539, 1540, 1541, 1550, 1551, 1552, 1553, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1664, 1665, 1718, 1719, 1720, 1721, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1752, 1753, 1754, 1755, 1780, 1781, 1782, 1783]

    # train_data = []
    # val_data = []
    # test_data = []

    # for i in list1:
    #     train_data.append(combined[i])
    #     # train_data.append(combined[i + 1])
    # for j in list2:
    #     val_data.append(combined[j])
    #     # val_data.append(combined[j + 1])
    # for k in list3:
    #     test_data.append(combined[k])
    #     # test_data.append(combined[k + 1])

    with open('test_data_list_total.json', 'w') as json_file:
    # 使用json.dump()将列表写入文件
        json.dump(test_data, json_file)
    for one in train_data[0 : 5]:
        print(one)

    return train_data, val_data, test_data


def create_dataloader(data, transform, batch_size = 32, shuffle=True):
    image_paths, target_paths, text_descriptions = zip(*data)
    dataset = SaliencyDatasetWithText(list(image_paths), list(target_paths), list(text_descriptions), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


