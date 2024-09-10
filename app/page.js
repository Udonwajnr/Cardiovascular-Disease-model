"use client"
import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import Papa from 'papaparse';

export default function Home() {
  const [model, setModel] = useState(null);
  const [formData, setFormData] = useState({
    age: '',
    sex: '',
    chestPainType: '',
    restingBP: '',
    cholesterol: '',
    fastingBS: '',
    restingECG: '',
    maxHR: '',
    exerciseAngina: '',
    oldpeak: '',
    stSlope: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [trainingAccuracy, setTrainingAccuracy] = useState([]);

  // Load and preprocess the data
  useEffect(() => {
    const loadData = async () => {
      const response = await fetch('/heart.csv');
      const reader = response.body.getReader();
      const result = await reader.read(); 
      const decoder = new TextDecoder('utf-8');
      const csv = decoder.decode(result.value); 
      const data = Papa.parse(csv, { header: true }).data; 
      const processedData = preprocessData(data);
      trainModel(processedData);
    };

    loadData();
  }, []);

  const preprocessData = (data) => {
    const sexMapping = { 'M': 1, 'F': 0 };
    const chestPainMapping = { 'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3 };
    const restingECGMapping = { 'Normal': 0, 'ST': 1, 'LVH': 2 };
    const exerciseAnginaMapping = { 'Y': 1, 'N': 0 };
    const stSlopeMapping = { 'Up': 0, 'Flat': 1, 'Down': 2 };

    const features = data.map(item => [
      parseFloat(item.Age),
      sexMapping[item.Sex],
      chestPainMapping[item.ChestPainType],
      parseFloat(item.RestingBP),
      parseFloat(item.Cholesterol),
      parseFloat(item.FastingBS),
      restingECGMapping[item.RestingECG],
      parseFloat(item.MaxHR),
      exerciseAnginaMapping[item.ExerciseAngina],
      parseFloat(item.Oldpeak),
      stSlopeMapping[item.ST_Slope]
    ]);

    const labels = data.map(item => parseFloat(item.HeartDisease));

    return { features, labels };
  };

  const createModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [11] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
      optimizer: tf.train.adam(),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    return model;
  };

  const trainModel = async (data) => {
    const { features, labels } = data;
    const xs = tf.tensor2d(features);
    const ys = tf.tensor2d(labels, [labels.length, 1]);

    const model = createModel();
    const history = await model.fit(xs, ys, {
      epochs: 50,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10 })
    });

    setModel(model);
    // Convert accuracy to percentage
    const accuracyInPercent = history.history.acc.map(acc => acc * 100);
    setTrainingAccuracy(accuracyInPercent);
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!model) {
      alert('Model is still training, please wait.');
      return;
    }

    const input = [
      parseFloat(formData.age),
      parseFloat(formData.sex),
      parseFloat(formData.chestPainType),
      parseFloat(formData.restingBP),
      parseFloat(formData.cholesterol),
      parseFloat(formData.fastingBS),
      parseFloat(formData.restingECG),
      parseFloat(formData.maxHR),
      parseFloat(formData.exerciseAngina),
      parseFloat(formData.oldpeak),
      parseFloat(formData.stSlope)
    ];

    const inputTensor = tf.tensor2d([input]);
    const prediction = model.predict(inputTensor);
    const predictionResult = prediction.dataSync()[0];
    setPrediction(predictionResult > 0.5 ? 'Positive for Cardiovascular Disease' : 'Negative for Cardiovascular Disease');
  };


  return (
    <div className="min-h-screen bg-gray-200 flex items-center justify-center">
  <div className="bg-white p-10 rounded-lg shadow-lg max-w-md w-full border border-gray-300">
    <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">Cardiovascular Disease Prediction</h1>
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <input
          name="age"
          type="number"
          placeholder="Age"
          value={formData.age}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div>
        <select
          name="sex"
          value={formData.sex}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select Sex</option>
          <option value="1">Male</option>
          <option value="0">Female</option>
        </select>
      </div>
      <div>
        <select
          name="chestPainType"
          value={formData.chestPainType}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select Chest Pain Type</option>
          <option value="0">ATA</option>
          <option value="1">NAP</option>
          <option value="2">ASY</option>
          <option value="3">TA</option>
        </select>
      </div>
      <div>
        <input
          name="restingBP"
          type="number"
          placeholder="Resting BP"
          value={formData.restingBP}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div>
        <input
          name="cholesterol"
          type="number"
          placeholder="Cholesterol"
          value={formData.cholesterol}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div>
        <input
          name="fastingBS"
          type="number"
          placeholder="Fasting Blood Sugar"
          value={formData.fastingBS}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div>
        <select
          name="restingECG"
          value={formData.restingECG}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select Resting ECG</option>
          <option value="0">Normal</option>
          <option value="1">ST</option>
          <option value="2">LVH</option>
        </select>
      </div>
      <div>
        <input
          name="maxHR"
          type="number"
          placeholder="Max HR"
          value={formData.maxHR}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div>
        <select
          name="exerciseAngina"
          value={formData.exerciseAngina}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Exercise Induced Angina</option>
          <option value="1">Yes</option>
          <option value="0">No</option>
        </select>
      </div>
      <div>
        <input
          name="oldpeak"
          type="number"
          placeholder="Oldpeak"
          value={formData.oldpeak}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div>
        <select
          name="stSlope"
          value={formData.stSlope}
          onChange={handleChange}
          required
          className="w-full p-4 border border-gray-300 rounded-lg text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select ST Slope</option>
          <option value="0">Up</option>
          <option value="1">Flat</option>
          <option value="2">Down</option>
        </select>
      </div>
      <div>
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition duration-200 text-lg"
        >
          Predict
        </button>
      </div>
    </form>
    {prediction && (
      <div className="mt-8 text-center">
        <p className="text-xl font-semibold text-gray-800">Prediction: {prediction}</p>
      </div>
    )}
    {trainingAccuracy && (
      <div className="mt-8 text-center">
        <button
          className="w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-700 transition duration-200 text-lg"
          onClick={() => alert(`Training Accuracy: ${trainingAccuracy[trainingAccuracy.length - 1].toFixed(2)}%`)}
        >
          Show Training Accuracy
        </button>
      </div>
    )}
  </div>
</div>

  

  );
}
