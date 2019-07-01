class generate_Dataset:
    def __init__(self,path):  # path ： such as  begn_CFG.txt
        self.maxLen_ins = 0   #  max instruction length
        self.maxLen_gadget = 0# max Gadget length
        self.count_gadget = 0   # Gadget length
        self.num_gadget = 0     # Gadget number
        self.max_gadget_byte = 0  # byte length of max gadget
        self.path = path
        self.raw_data = []

        self.get_raw_data(path)
        self.begnData = self.generate_begnData()
        self.negData = self.generate_negData()


    def get_raw_data(self,path):

        max_ins = ""  # max length instruction
        max_gadget = 0 # max length position
        l = 0  #max length ins position

        max_gadget_chain_len = 0
        max_gadget_chain_len_byte = 0

        f = open(self.path,'r')
        cur_len = 0
        chain_len = 0
        num = 0
        flag = 0
        for line in f.readlines():
            num = num + 1
            cur_len += len(line[:-1])
            if len(line[:-1]) > self.maxLen_ins :
                self.maxLen_ins = len(line[:-1])
                max_ins = line[:-1]
                l = num
            self.raw_data.append(line[:-1])
            self.count_gadget = self.count_gadget + 1
            if line == "\n" :
                if self.count_gadget > self.maxLen_gadget :
                    self.maxLen_gadget = self.count_gadget
                    max_gadget = num
                    self.max_gadget_byte = cur_len
                self.num_gadget += 1
                self.count_gadget = 0
                cur_len = 0
                flag = flag + 1
                chain_len += cur_len

        f.close()

        print("num of Gadget : %d" %(self.num_gadget))
        print("max ins length : %d" %(self.maxLen_ins))
        print("max length ins : %s, in line : %d" %(max_ins,l))
        print("max gadget length : %d" %(self.maxLen_gadget-1))
        print("max gadget position : %d" %(max_gadget))
        print("byte of max gadget : %d" %(self.max_gadget_byte))
        #for i in range(20):
         #   print(self.raw_data[i])

    #data procress
    #
    def generate_begnData(self):
        flag = 0
        begnData = []
        last_tmp = ""
        tmp = ""
        for line in self.raw_data :
            if line != "" :
                tmp += line
            if line == "" :
                for i in range(int((self.max_gadget_byte - len(tmp)) / 2)):
                    tmp += "90"
                flag += 1
                if last_tmp == "":
                    last_tmp = tmp
                    print("length of each data : %d" %(len(tmp)))
            #if flag == 2 :
                t = tmp
                tmp = last_tmp + tmp
                last_tmp = t
                if len(tmp) != 2 * self.max_gadget_byte :
                    tmp += "90"
                flag = 0
                begnData.append(tmp)
                tmp = ""
        print("benign dataset number: %d" %(len(begnData)))
        print("example : %s" %(begnData[:1]))
        return begnData

    def generate_negData(self):
        tmp = ""
        Gadget = []
        negData = []
        cnt= 0
        for line in self.raw_data :
            cnt +=1
            if line != "" :
                tmp += line
            if line == "" :
                for i in range(int((self.max_gadget_byte - len(tmp)) / 2)):
                    tmp += "90"
                Gadget.append(tmp)  #Gadget
                tmp = ""
        L = len(Gadget)
        print("gadget number : %d " %(L))
        for i in range(int(L/2)):
            negData.append(Gadget[i]+Gadget[i+int(L/2)])#gadget chain
        for i in range(int(L/3)):
            negData.append(Gadget[int(L/3)-i]+Gadget[i+int(2*L/3)])
        print("negative gadget chain number(DataSet) : %d" % (len(negData)))
        print("example : %s" %(negData[:1]))
        return negData

    def getData(self):
        x1 = []
        x2 = []
        for i in self.begnData:
            for j in range(2 * self.max_gadget_byte-10):
            #print ("len i",len(i),"2 * max gadget length - 10 :",2 * self.max_gadget_byte-10)
            #for j in range(len(i)):
                x1.append(float(int(i[j],16)))

        for i in self.negData:
            for j in range(2 * self.max_gadget_byte-10):
                x2.append(float(int(i[j],16)))
        print("benign data:",x1[:2 * self.max_gadget_byte-10])
        print("negdata data:", x2[:2 * self.max_gadget_byte-10])
        return x1,x2
        #return self.begnData,self.negData
    def getMaxGadget(self):  # gadget chain length
        return 2 * self.max_gadget_byte

if __name__ == '__main__':
    a = generate_Dataset("begn_CFG.txt")
    a.getData()
