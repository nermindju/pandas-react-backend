// Define the request payload with only type and year
const data = {
  type: 'SUV',
  year: 2022
};

// Make the fetch request to the Flask server
fetch('http://127.0.0.1:5000/get_XGBoost_prediction', {
  method: 'POST',
  headers: {
      'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(data => {
  if (data.error) {
      console.error('Error:', data.error);
  } else {
      console.log('Prediction:', data.prediction);
      console.log('Vehicles:', data.vehicles);
  }
})
.catch(error => {
  console.error('Fetch Error:', error);
});


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




  
