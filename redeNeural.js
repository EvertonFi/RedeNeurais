function sigmoid(x) {
  return 1/(1 + Math.exp(-x));
}

function dsigmoid(x) {
  return x * (1-x);
}

class RedeNeural{
  constructor(input, hidden, output){
    this.input = input;
    this.hidden = hidden;
    this.output = output;

    this.bias_ih = new Matrix(this.hidden,1);
    this.bias_ih.randomize();
    this.bias_ho = new Matrix(this.output,1);
    this.bias_ho.randomize();
    
    this.weights_ih = new Matrix(this.hidden, this.input);
    this.weights_ih.randomize();

    this.weights_ho = new Matrix(this.output, this.hidden);
    this.weights_ho.randomize();

    this.learning_rate = 0.1;

  }

  train(arr, target) {
    //Input Hidden
    let input = Matrix.arrayToMatrix(arr);
        
    let hidden = Matrix.multiply(this.weights_ih,input);
    hidden = Matrix.add(hidden,this.bias_ih);

    hidden.map(sigmoid);

    //hidden output

    let output = Matrix.multiply(this.weights_ho, hidden);
    output = Matrix.add(output,this.bias_ho);
    output.map(sigmoid);

    //BackPropagation
    let expected = Matrix.arrayToMatrix(target);
    let output_error = Matrix.subtract(expected, output);
    let d_output = Matrix.map(output, dsigmoid);

    let hidden_T = Matrix.transpose(hidden);

    let gradient = Matrix.hadamard(d_output, output_error);
    gradient = Matrix.escalar_multiply(gradient, this.learning_rate)
  
    gradient = Matrix.multiply(gradient,hidden_T);
    gradient.print();
  }
}