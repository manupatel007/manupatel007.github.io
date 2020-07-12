let mobilenet;
let model;
var webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var maxm=0;
let isPredicting = false;

var arr = [0,0,0,0,0,0,0,0,0,0];

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  maxm++; //as training starts from zero
  dataset.ys = null;
  dataset.encodeLabels(maxm);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: maxm, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
  document.getElementById("dummy").innerText = "Training is complete!";
  window.alert("Training is complete!")
}


function handleButton(elem){
	arr[elem.id]++;
	document.getElementById(String(elem.id)+String(elem.id)).innerText = "Class "+elem.id+" Samples:" + arr[elem.id];
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);
	if(maxm<elem.id){
		maxm=elem.id;
	}
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    predictionText="I see class " + classId;
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	webcam = new Webcam(document.getElementById('wc'));
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}

async function init1(){
	webcam = new Webcam(document.getElementById('wc'));
	await webcam.setup();
	}

async function dwnld(){
	await model.save('downloads://my-model');
}


init();