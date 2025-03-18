function checkSpam() {
    const text = document.getElementById("textInput").value;
    if (!text) {
        alert("Please enter some text.");
        return;
    }

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = `Prediction: ${data.prediction}`;
    })
    .catch(error => console.error("Error:", error));
}
