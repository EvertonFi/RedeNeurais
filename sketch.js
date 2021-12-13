function setup(){
  createCanvas(200,200);
  background(0);

  nn = new RedeNeural(2,3,2);
  let arr = [1,2];
  nn.train(arr,[0,1]);

}

function draw(){

}