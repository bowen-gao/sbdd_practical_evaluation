from matplotlib import pyplot as plt



seq_ef = [23.83, 8.59]

seq_x = [16569, 11664]



struct_ef = [19.99, 8.85, 4.90]
strut_x = [17519,13537, 11868]


plt.scatter(seq_x, seq_ef, label='Sequential')

plt.scatter(strut_x, struct_ef, label='Structural')


#plt.plot(strut_x, struct_ef, label='Structural')

# show legend

plt.legend()


plt.ylabel('EF')
plt.xlabel('number of training data')




plt.savefig('ef.png')


