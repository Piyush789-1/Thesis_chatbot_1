import streamlit as st
from streamlit_chat import message
from model_loader import answer_query  # Your answer_query function
from PIL import Image
import re
import os

# -------------------------------
# Step 2: Initialize the conversation history in session state.
# -------------------------------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # Initialize as empty list

# -------------------------------
# Define the display_rich_message function (no chain-of-thought toggle)
# -------------------------------
def display_rich_message(raw_message: str):
    """
    Displays rich bot messages that may include markdown text and images.
    
    The LLM output "raw_message" contains a marker "Final Answer:".
    This function splits on that marker and shows only the text after it.
    """
    # If the marker exists, show only the text after it.
    
    if "Final Answer:" in raw_message:
        parts = raw_message.split("Final Answer:")
        message_text = parts[-1].strip()
    else:
        message_text = raw_message

    # Process image markers if any, using regex.
    image_pattern = r"!\[IMG_TITLE:([^\]]+)\]\(([^)]+)\)"
    segments = []
    last_idx = 0

    for match in re.finditer(image_pattern, message_text):
        start, end = match.span()
        caption = match.group(1).strip()
        file_path = match.group(2).strip().replace("\\", "/")
        if start > last_idx:
            segments.append(('text', message_text[last_idx:start]))
        segments.append(('image', file_path, caption))
        last_idx = end

    if last_idx < len(message_text):
        segments.append(('text', message_text[last_idx:]))

    # Render each segment.
    for seg in segments:
        if seg[0] == 'text':
            st.markdown(seg[1])
        elif seg[0] == 'image':
            file_path, caption = seg[1], seg[2]
            st.markdown(f"**{caption}**")
            if os.path.exists(file_path):
                try:
                    img = Image.open(file_path)
                    st.image(img)
                except Exception as e:
                    st.error("Error loading image: " + str(e))
            else:
                st.markdown(f"*Image not found at:* `{file_path}`")

# -------------------------------
# Main UI Layout
# -------------------------------
st.title("Chat with Thesis")
st.write("This interface shows the conversation as a thread. The new question field is always at the bottom.")

# Remove the chain-of-thought toggle by not including a checkbox.

# -------------------------------
# Provide a Clear Conversation History button.
# -------------------------------
if st.button("Clear Conversation History"):
    st.session_state.conversation_history = []  # Clear the conversation history
    

# Render the conversation thread.
for i, chat in enumerate(st.session_state.conversation_history):
    if chat["role"] == "user":
        message(chat["message"], is_user=True, key=f"user_{i}")
    else:
        st.markdown("**Bot:**")
        display_rich_message(chat["message"])
        st.markdown("---")

# -------------------------------
# New question input area as an st.form.
# -------------------------------
with st.form(key="chat_form", clear_on_submit=True):
    user_question = st.text_input("Your Question:")
    submit_button = st.form_submit_button("Send")
    if submit_button and user_question.strip():
        # Append the user's question to the conversation thread.
        st.session_state.conversation_history.append({
            "role": "user",
            "message": user_question
        })
        # Get the bot's response.
        response = answer_query(user_question)
        if isinstance(response, dict):
            response_text = response.get("choices", [{}])[0].get("text", "")
        else:
            response_text = response
        # Append the bot's answer.
        st.session_state.conversation_history.append({
            "role": "bot",
            "message": response_text
        })
