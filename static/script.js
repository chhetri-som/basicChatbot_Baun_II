// Function to send a message to the chatbot
function sendMessage() {
    const userInput = document.getElementById("user-input");
    const userMessage = userInput.value.trim();

    if (userMessage === "") {
        return;
    }

    // Display user message
    displayMessage(userMessage, "user-message");

    // Clear input box
    userInput.value = "";

    // Send message to Flask backend
    fetch("/get_response", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userMessage }),
    })
    .then(response => response.json())
    .then(data => {
        // Display bot response
        displayMessage(data.response, "bot-message");
    })
    .catch(error => {
        console.error("Error:", error);
        displayMessage("An error occurred. Please try again.", "bot-message");
    });
}

// Function to display messages in the chat box
function displayMessage(message, messageType) {
    const chatBox = document.getElementById("chat-box");
    const messageElement = document.createElement("div");
    messageElement.classList.add("message", messageType);
    messageElement.innerHTML = `<p>${message}</p>`;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Allow sending message by pressing Enter
document.getElementById("user-input").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});