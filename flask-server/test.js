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
//   console.log('Response:', data.message);  // Log the message from the server
// })
// .catch((error) => {
//   console.error('Fetch Error:', error);
// });

// // Example to select 'volkswagen' as the brand
// fetch('http://127.0.0.1:5000/select_brand', {
//   method: 'POST',
//   headers: {
//       'Content-Type': 'application/json'
//   },
//   body: JSON.stringify({ brand: 'audi' })
// })
// .then(response => response.json())
// .then(data => console.log(data))
// .catch(error => console.error('Error:', error));



// // // Model choice to send
// const modelChoice = 'random_forest';  // or 'random_forest'

// // Sending the POST request
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




  
