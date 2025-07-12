import os
from mistralai import Mistral
import json
from typing import List, Dict, Any
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import datetime

class JupyterChatInterface:
    """Enhanced chat interface for Jupyter notebook"""
    
    def __init__(self):
        # Create UI components
        self.chat_output = widgets.Output(layout={'height': '400px', 'overflow_y': 'auto'})
        self.input_text = widgets.Text(
            placeholder='Type your message here...',
            layout=widgets.Layout(width='80%'),
            style={'description_width': 'initial'}
        )
        self.send_button = widgets.Button(
            description='Send',
            button_style='primary',
            layout=widgets.Layout(width='18%')
        )
        self.clear_button = widgets.Button(
            description='Clear Chat',
            button_style='warning',
            layout=widgets.Layout(width='100px')
        )
        
        # Create input row
        self.input_row = widgets.HBox([self.input_text, self.send_button])
        
        # Create main container
        self.chat_container = widgets.VBox([
            widgets.HTML("<h3>🌦️ Weather Assistant</h3>"),
            self.chat_output,
            self.input_row,
            self.clear_button
        ])
        
        # Setup event handlers
        self.send_button.on_click(self._on_send_click)
        self.input_text.on_submit(self._on_input_submit)
        self.clear_button.on_click(self._on_clear_click)
        
        # Callback for when user sends a message
        self.on_message_callback = None
        
    def display(self):
        """Display the chat interface"""
        display(self.chat_container)
        self.display_message("Hello! I'm your weather assistant. Ask me about the weather in any city!", "Assistant")
        
    def _on_send_click(self, button):
        """Handle send button click"""
        self._send_message()
        
    def _on_input_submit(self, text_widget):
        """Handle input text submission (Enter key)"""
        self._send_message()
        
    def _on_clear_click(self, button):
        """Handle clear button click"""
        with self.chat_output:
            clear_output()
        
    def _send_message(self):
        """Send the current message"""
        message = self.input_text.value.strip()
        if message and self.on_message_callback:
            self.input_text.value = ""  # Clear input
            self.display_message(message, "You")
            self.on_message_callback(message)
    
    def display_message(self, message: str, sender: str = "Assistant"):
        """Display a message in the chat"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if sender == "You":
            color = "#0066cc"
            align = "right"
        else:
            color = "#009900"
            align = "left"
            
        html_message = f"""
        <div style="margin: 5px 0; text-align: {align};">
            <div style="display: inline-block; max-width: 80%; padding: 8px 12px; 
                       border-radius: 10px; background-color: {color}; color: white;">
                <strong>{sender}</strong> <span style="font-size: 0.8em; opacity: 0.7;">{timestamp}</span><br>
                {message}
            </div>
        </div>
        """
        
        with self.chat_output:
            display(HTML(html_message))
            
    def show_typing_indicator(self):
        """Show typing indicator"""
        with self.chat_output:
            display(HTML("""
            <div style="margin: 5px 0; text-align: left;">
                <div style="display: inline-block; padding: 8px 12px; 
                           border-radius: 10px; background-color: #f0f0f0; color: #666;">
                    <em>Assistant is typing...</em>
                </div>
            </div>
            """))

class ChatAssistantUI:
    """Enhanced Chat assistant with Jupyter UI"""
    
    def __init__(self, tools, developer_prompt: str, api_key: str = None):
        self.tools = tools
        self.developer_prompt = developer_prompt
        
        # Initialize Mistral client
        if not api_key:
            api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Please provide MISTRAL_API_KEY")
        
        self.mistral_client = Mistral(api_key=api_key)
        self.conversation_history = []
        
        # Add system message
        self.conversation_history.append({
            "role": "system",
            "content": developer_prompt
        })
        
        # Create UI
        self.chat_interface = JupyterChatInterface()
        self.chat_interface.on_message_callback = self._handle_user_message
        
    def _convert_mcp_tools_to_mistral_format(self) -> List[Dict]:
        """Convert MCP tools to Mistral function calling format"""
        mistral_tools = []
        
        try:
            # Get tools from MCP client
            tools_list = self.tools.mcp_client.get_tools()
            
            for tool in tools_list.get('tools', []):
                mistral_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool.get("inputSchema", {})
                    }
                }
                mistral_tools.append(mistral_tool)
        except Exception as e:
            print(f"Error getting tools: {e}")
            
        return mistral_tools
    
    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call MCP tool and return result"""
        try:
            result = self.tools.mcp_client.call_tool(tool_name, arguments)
            return str(result)
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"
    
    def _handle_user_message(self, message: str):
        """Handle user message asynchronously"""
        # Show typing indicator
        self.chat_interface.show_typing_indicator()
        
        try:
            response = self._get_mistral_response(message)
            
            # Clear typing indicator and show response
            with self.chat_interface.chat_output:
                # Remove the last message (typing indicator)
                clear_output(wait=True)
                
            # Re-display all messages except the typing indicator
            self._redisplay_conversation()
            
            self.chat_interface.display_message(response, "Assistant")
            
        except Exception as e:
            with self.chat_interface.chat_output:
                clear_output(wait=True)
            self._redisplay_conversation()
            self.chat_interface.display_message(f"Error: {str(e)}", "Assistant")
    
    def _redisplay_conversation(self):
        """Redisplay the conversation history"""
        for msg in self.conversation_history:
            if msg["role"] == "user":
                self.chat_interface.display_message(msg["content"], "You")
            elif msg["role"] == "assistant" and "content" in msg and msg["content"]:
                self.chat_interface.display_message(msg["content"], "Assistant")
    
    def _get_mistral_response(self, message: str) -> str:
        """Get response from Mistral AI"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Get available tools
        tools = self._convert_mcp_tools_to_mistral_format()
        
        try:
            # Call Mistral API
            response = self.mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=self.conversation_history,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None
            )
            
            assistant_message = response.choices[0].message
            
            # Handle tool calls
            if assistant_message.tool_calls:
                # Add assistant message with tool calls to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        for tool_call in assistant_message.tool_calls
                    ]
                })
                
                # Execute tool calls
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Call the tool
                    tool_result = self._call_tool(function_name, function_args)
                    
                    # Add tool result to history
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                # Get final response after tool execution
                final_response = self.mistral_client.chat.complete(
                    model="mistral-large-latest",
                    messages=self.conversation_history
                )
                
                final_content = final_response.choices[0].message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_content
                })
                
                return final_content
            else:
                # No tool calls, just return the response
                content = assistant_message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": content
                })
                return content
                
        except Exception as e:
            return f"Error getting response from Mistral: {str(e)}"
    
    def start(self):
        """Start the chat interface"""
        self.chat_interface.display()