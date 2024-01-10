"Function that sorts numbers given by the user"

def main():
    a = input("Give a list of integers separated by space: ")
    numbers = list(map(int, a.split()))
    numbers.sort()
    print("Given numbers sorted: ", numbers)

main()
