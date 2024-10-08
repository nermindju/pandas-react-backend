// // Define the request payload with only type and year
// const data = {
//   type: 'SUV',
//   year: 2022
// };

// // Make the fetch request to the Flask server
// fetch('http://127.0.0.1:5000/get_prediction', {
//   method: 'POST',
//   headers: {
//       'Content-Type': 'application/json'
//   },
//   body: JSON.stringify(data)
// })
// .then(response => response.json())
// .then(data => {
//   if (data.error) {
//       console.error('Error:', data.error);
//   } else {
//       console.log('Prediction:', data.prediction);
//       console.log('Vehicles:', data.vehicles);
//       console.log('Metrics:', data.model_metrics)
//   }
// })
// .catch(error => {
//   console.error('Fetch Error:', error);
// });

// fetch('http://127.0.0.1:5000/train_model', {
//   method: 'POST',
//   headers: {
//       'Content-Type': 'application/json',
//   },
//   body: JSON.stringify({
//       model: 'xgboost'
//   }),
// })
// .then(response => response.json())  // Parse the response as JSON
// .then(data => {
//   console.log('Response:', data.message, data.results);  // Log the message from the server
// })
// .catch((error) => {
//   console.error('Fetch Error:', error);
// });

// Function to fetch available models
// Function to fetch available models from the server
// function fetchModels(modelType) {
//   fetch(`http://127.0.0.1:5000/get_models?model_type=${modelType}`, {
//       method: 'GET',
//       headers: {
//           'Content-Type': 'application/json',
//       }
//   })
//   .then(response => {
//       if (!response.ok) {
//           throw new Error(`HTTP error! status: ${response.status}`);
//       }
//       return response.json();
//   })
//   .then(data => {
//       if (data.status === 'success') {
//           console.log(`Available ${modelType} models:`, data.models);
//       } else {
//           console.log(`Error:`, data.message);
//       }
//   })
//   .catch(error => {
//       console.error('Error fetching models:', error);
//   });
// }

// // // // Example usage to fetch new or old models
// fetchModels('old');  // To fetch new models
// fetchModels('new');  // To fetch new models


// // Function to set the selected model on the server
// function setSelectedModel(modelName, modelType) {
//   fetch('http://127.0.0.1:5000/set_selected_model', {
//       method: 'POST',
//       headers: {
//           'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({
//           model_name: modelName,
//           model_type: modelType
//       })
//   })
//   .then(response => {
//       if (!response.ok) {
//           throw new Error(`HTTP error! status: ${response.status}`);
//       }
//       return response.json();
//   })
//   .then(data => {
//       if (data.status === 'success') {
//           console.log('Model selected successfully:', data.message);
//           console.log('Model path:', data.model_path);
//       } else {
//           console.log('Error:', data.message);
//       }
//   })
//   .catch(error => {
//       console.error('Error setting model:', error);
//   });
// }

// // // // // // Example usage to set the selected model
// setSelectedModel('xgboost_model_20240909_131424.joblib', 'old');  // Set a model from old models




// Example to select 'volkswagen' as the brand
// fetch('http://127.0.0.1:5000/select_brand', {
//   method: 'POST',
//   headers: {
//       'Content-Type': 'application/json'
//   },
//   body: JSON.stringify({ brand: 'volkswagen' })
// })
// .then(response => response.json())
// .then(data => console.log(data))
// .catch(error => console.error('Error:', error));



// // Model choice to send
// const modelChoice = 'random_forest';  // or 'random_forest'

// // // Sending the POST request
// fetch('http://127.0.0.1:5000/set_model_choice', {
//     method: 'POST',
//     headers: {
//         'Content-Type': 'application/json',
//     },
//     body: JSON.stringify({ model_choice: modelChoice }),
// })
// .then(response => response.json())
// .then(data => {
//     console.log(data);
// })
// .catch(error => console.error('Error:', error));






// //Fetch the data from the Flask server
// fetch('http://localhost:5000/hist_plot')
//   .then(response => {
//     if (!response.ok) {
//       throw new Error('Network response was not ok');
//     }
//     return response.json();
//   })
//   .then(data => {
//     console.log('Price Range Data:', data);
//   })
//   .catch(error => {
//     console.error('There was a problem with the fetch operation:', error);
//   });


// fetch('http://localhost:5000/model_ranking')
//   .then(response => {
//     if (!response.ok) {
//       throw new Error('Network response was not ok');
//     }
//     return response.json();
//   })
//   .then(data => {
//     console.log('Top Car Models:', data);
//   })
//   .catch(error => {
//     console.error('There was a problem with the fetch operation:', error);
//   });

// fetch("http://localhost:5000/models_average_price")
//   .then((response) => {
//     if (!response.ok) {
//       throw new Error("Network response was not ok");
//     }
//     return response.json();
//   })
//   .then((data) => {
//     console.log("25 most popular models avg price:", data);
//   })
//   .catch((error) => {
//     console.error("There was a problem with the fetch operation:", error);
//   });


  // JavaScript code to test the "/get_models_price_box" route using fetch
// JavaScript code to test the "/get_models_price_box" route using fetch
// fetch('http://localhost:5000/get_models_price_box', {
//     method: 'GET',
//     headers: {
//         'Content-Type': 'application/json'
//     }
// })
// .then(response => {
//     if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//     }
//     return response.json();
// })
// .then(data => {
//     // Beautify the JSON output to see everything clearly
//     console.log('Response data:', JSON.stringify(data, null, 2));
// })
// .catch(error => {
//     console.error('There was an error!', error);
// });




  
