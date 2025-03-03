import math

class UserMatchcode(object):
    @classmethod
    def calculateNumber(cls, input1, input2):
        # Step 1: Square each element in input2
        squared_arr = [x**2 for x in input2]
        
        # Step 2: Find binary equivalent and count set bits
        set_bits = [bin(x).count('1') for x in squared_arr]
        
        # Step 3: Find X and Y
        max_bits = max(set_bits)
        min_bits = min(set_bits)
        
        X = min([x for x, bits in zip(squared_arr, set_bits) if bits == max_bits])
        Y = min([x for x, bits in zip(squared_arr, set_bits) if bits == min_bits])
        
        # Step 4: Multiply X and Y
        product = X * Y
        
        # Step 5: Find the nearest power of 2
        if product == 0:
            return 1
        power = math.ceil(math.log2(product))
        nearest_power = 2 ** power
        
        return nearest_power

# Example usage:
input1 = 1
input2 = [3, 5, 7]  
print(UserMatchcode.calculateNumber(input1, input2))  # Output: 256