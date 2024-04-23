import math 
def sigmnoid(x):
    o = 1/(1+math.e**-x)
    return o 

epoch = 5 
def bep(X,inputweights1,inputweights2,outputweights,  desired_outpu , e):
    while e <= epoch-1 :
        print('-' * 45 ,"epoch :" ,e+1 , '-' * 45)
        l , l2 , errorj = [] , [] , []
        errork = []
        a,b,c = 0,0,0
        for i , j , k ,in zip(X,inputweights1 , inputweights2):
            a += i*j
            b += i*k
        
        o5 = sigmnoid(a)
        o6 = sigmnoid(b)
        l.append(o5)
        l.append(o6)

        for i , j in zip(l+[1],outputweights) : 
            c += i*j

        o7 = sigmnoid(c)
        l2.append(o7)

        errorr = desired_outpu - o7
                
        print(f'o5 = {o5} \no6 = {o6} \no7 = {o7} \nError = {errorr}')

        for i in l2 : 
            errk = o7 * (desired_outpu-o7) * (1 - o7)
            errork.append(errk)
        
        sum_err = sum(errork)

        for i,j in zip(l,outputweights) :
            errj = i * (desired_outpu - i ) * sum_err * j
            errorj.append(errj)
        print(f'Error at outpur layer is :- {errork}')
        print(f'Errors at hidden layer are :-{errorj}')
        
        def update_new_weights(weights, error , learning_rate , oi , node):
            weight_list = []
            wij = 0
            for i , k in zip(weights ,oi):
                wij = i + (learning_rate * k * error[node-1])
                weight_list.append(wij)
            return weight_list
        
        new_weights_for_1st_node = update_new_weights(weights = inputweights1 ,
                                                    error = errorj, 
                                                    learning_rate = 0.8 ,
                                                    oi= X , 
                                                    node =1 )
        
        new_weights_for_2nd_node = update_new_weights(weights=inputweights2 , 
                                                    error = errorj , 
                                                    learning_rate= 0.8 , 
                                                  oi = X , 
                                                  node= 2)
    
        new_weights_for_output_node = update_new_weights(weights = outputweights , 
                                                        error = [sum_err] , 
                                                        learning_rate= 0.8 , 
                                                        oi = l+[1] , 
                                                        node = 1)
        
        print(f'New Weights at Hidden Layer :- {new_weights_for_1st_node,new_weights_for_2nd_node} \nNew Weights at Output Layer = {new_weights_for_output_node}\n')

        return bep(X , 
            new_weights_for_1st_node ,
            new_weights_for_2nd_node ,
            new_weights_for_output_node , 
            desired_outpu ,
            e+1
            )

# --------------------------------------------- epoch : 1 ---------------------------------------------
# o5 = 0.598687660112452 
# o6 = 0.7310585786300049 
# o7 = 0.4174148989550627 
# Error = 0.5825851010449373
# Error at outpur layer is :- [0.14167287072891396]
# Errors at hidden layer are :-[-0.010211528871801637, 0.005570915400376464]
# New Weights at Hidden Layer :- ([0.29183077690255865, -0.20816922309744132, 0.1, 0.0918307769025587, 0.1918307769025587], [0.10445673232030117, 0.4044567323203012, -0.3, 0.4044567323203012, 0.10445673232030117]) 
# New Weights at Output Layer = [-0.23214576041751406, 0.2828569340044098, -0.18666170341686883]

# --------------------------------------------- epoch : 2 ---------------------------------------------
# o5 = 0.5908119888434402 
# o6 = 0.7345490955539927 
# o7 = 0.47102153762087023 
# Error = 0.5289784623791298
# Error at outpur layer is :- [0.13180040525292397]
# Errors at hidden layer are :-[-0.007396899236691633, 0.007269233392444831]
# New Weights at Hidden Layer :- ([0.28591325751320534, -0.21408674248679463, 0.1, 0.08591325751320539, 0.1859132575132054], [0.11027211903425704, 0.4102721190342571, -0.3, 0.4102721190342571, 0.11027211903425704]) 
# New Weights at Output Layer = [-0.16985035277123292, 0.36030802878215784, -0.08122137921452965]

# --------------------------------------------- epoch : 3 ---------------------------------------------
# o5 = 0.5850776137985702 
# o6 = 0.7390599736805298 
# o7 = 0.5214099562015255 
# Error = 0.4785900437984745
# Error at outpur layer is :- [0.11942813186633403]
# Errors at hidden layer are :-[-0.0049244013366397015, 0.00829852607062217]
# New Weights at Hidden Layer :- ([0.28197373644389356, -0.21802626355610638, 0.1, 0.08197373644389362, 0.18197373644389364], [0.11691093989075478, 0.41691093989075484, -0.3, 0.41691093989075484, 0.11691093989075478]) 
# New Weights at Output Layer = [-0.11395057164101237, 0.430919670377236, 0.014321126278537571]

# --------------------------------------------- epoch : 4 ---------------------------------------------
# o5 = 0.5812470967438573 
# o6 = 0.7441485658595924 
# o7 = 0.5667874650979835 
# Error = 0.4332125349020165
# Error at outpur layer is :- [0.10637076084065905]
# Errors at hidden layer are :-[-0.0029502403707565884, 0.008727022374376907]
# New Weights at Hidden Layer :- ([0.2796135441472883, -0.22038645585271166, 0.1, 0.07961354414728836, 0.17961354414728836], [0.1238925577902563, 0.42389255779025636, -0.3, 0.42389255779025636, 0.1238925577902563]) 
# New Weights at Output Layer = [-0.06448841490735775, 0.4942441896804121, 0.09941773495106482]

# --------------------------------------------- epoch : 5 ---------------------------------------------
# o5 = 0.5789474770122246 
# o6 = 0.7494291805412191 
# o7 = 0.6064665128046811 
# Error = 0.3935334871953189
# Error at outpur layer is :- [0.09392262314725901]
# Errors at hidden layer are :-[-0.001476479275600441, 0.008717117066402663]
# New Weights at Hidden Layer :- ([0.27843236072680794, -0.221567639273192, 0.1, 0.07843236072680801, 0.17843236072680801], [0.13086625144337843, 0.4308662514433785, -0.3, 0.4308662514433785, 0.13086625144337843]) 
# New Weights at Output Layer = [-0.020987402342977295, 0.5505548732800377, 0.17455583346887205]


                                                     
