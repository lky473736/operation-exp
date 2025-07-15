# operation
(project) A regression machine that learns a dataset based on an equation created using various operators

## Introduction
This project started with the question of **whether AI can be given an intuition about formulas**. It is an experiment to determine whether AI can infer a formula given only specific terms and results. The key to the formula is understanding the relationship between operands (operators) and using this to predict results from given operands. This experiment aims to explore whether AI can learn rules between operands without being given operators and, furthermore, if it can infer the operators themselves.

Let's first predict X using the data below.

| operand1 | operand2 | result |
|----------|----------|--------|
|    1     |    2     |    3   |
|    2     |    3     |    5   |
|    3     |    4     |    7   |
|    4     |    5     |    **x**   |

From this, we see `result = operand1 + operand2`, so x can be predicted as 9. This prediction process signifies the ability to infer without being given the operator, based on the relationship between the operands. The thought process is summarized as follows:

1. **How to operate** on each operand to achieve the result? (Inferring the operator)
2. Try different operands to construct a general rule (operator).
3. Combine the constructed operator with operands to finally derive the value of x.

The core of this project is to investigate whether AI can learn this way of thinking. Specifically, we will experiment with whether a model trained on a dataframe with only operands (and no operators) can extract patterns.

## Object
The project has two main objectives:
1. **Predicting Results**: Can AI predict the result using only operands, without given operators, by learning patterns between the operands?
2. **Predicting Operators**: Can AI infer the relationship, i.e., the operator, between the given operands and result?

## Operators Used

### Arithmetic Operators
- `+` : Addition
- `-` : Subtraction
- `*` : Multiplication
- `/` : Division
- `%` : Modulus
- `**` : Exponentiation
- `//` : Integer Division (Floor Division)

### Comparison Operators
- `==` : Equals
- `!=` : Not Equals
- `>` : Greater Than
- `<` : Less Than
- `>=` : Greater Than or Equal To
- `<=` : Less Than or Equal To

### Bitwise Operators
- `&` : Bitwise AND
- `|` : Bitwise OR
- `^` : Bitwise XOR (Exclusive OR)
- `<<` : Bitwise Left Shift
- `>>` : Bitwise Right Shift

### Special Arithmetic Operators
- `divmod(a, b)` : Returns the quotient and remainder of dividing a by b
- `max(a, b)` : Maximum of two numbers
- `min(a, b)` : Minimum of two numbers
- `pow(a, b)` : a raised to the power of b
- `hypot(a, b)` : Euclidean distance between two numbers (Pythagorean Theorem)

### Miscellaneous Mathematical Operators
- `gcd(a, b)` : Greatest Common Divisor
- `lcm(a, b)` : Least Common Multiple
- `log(a, b)` : Logarithm of a with base b
- `mod(a, b)` : a mod b
- `abs(a - b)` : Absolute difference

## Experiment Sections

The experiments will be conducted in the following two sections.

### Section 1. Predict the Result without Operator
- Given n operands as features and the calculated result as the target.
- **Operators are not given**, and the trained model must predict the result based on operands only.
- Data will be prepared for each operator and file according to the number of operands (2 to 5 operands).
- Example: If 1, 2, 3 are features and 6 is the target, then the model should output 9 when given 2, 3, 4 (addition operation).
- This section tests whether AI can learn the relationship between operands and infer the result in given situations.

### Section 2. Predict the Operator
- Given n operands and the result as features, the operator is the target.
- **The trained model must receive operands and result as input and predict the corresponding operator** (approaching as a classification problem).
- This involves classifying which operator fits the given operands and result among various operators.
- For instance, with 1, 2, 3 as features and the operator being +, the model should learn this rule and predict the operator '+' when given 2, 3, 5.

## Report
After the experiments are completed, a detailed **report** will be written. The report will cover how AI learned the relationships between operands, prediction performance, and accuracy for each operator. The goal is to develop an AI model that infers mathematical rules with only operands and results without given operators.