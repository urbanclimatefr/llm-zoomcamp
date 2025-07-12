import os
from mistralai import Mistral
import json
from typing import List, Dict, Any

class ChatInterface:
    """Simple chat interface for user interaction"""
    
    def get_user_input(self) -> str:
        return input("You: ")
    
    def display_message(self, message: str, sender: str = "Assistant"):
        print(f"{sender}: {message}")

class ChatAssistant:
    """Chat assistant using Mistral AI with MCP tools"""
    
    def __init__(self, tools, developer_prompt: str, chat_interface: ChatInterface, client=None):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface
        
        # Initialize Mistral client
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Please set MISTRAL_API_KEY environment variable")
        
        self.mistral_client = Mistral(api_key=api_key)
        self.conversation_history = []
        
        # Add system message
        self.conversation_history.append({
            "role": "system",
            "content": developer_prompt
        })
    
    def _convert_mcp_tools_to_mistral_format(self) -> List[Dict]:
        """Convert MCP tools to Mistral function calling format"""
        mistral_tools = []
        
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
        
        return mistral_tools
    
    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call MCP tool and return result"""
        try:
            result = self.tools.mcp_client.call_tool(tool_name, arguments)
            return str(result)
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"
    
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
                model="mistral-large-latest",  # or "mistral-small-latest"
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
    
    def run(self):
        """Start the chat loop"""
        self.chat_interface.display_message("Hello! I'm your weather assistant. Ask me about the weather in any city!")
        self.chat_interface.display_message("Type 'quit' to exit.")
        
        while True:
            try:
                user_input = self.chat_interface.get_user_input()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    self.chat_interface.display_message("Goodbye!")
                    break
                
                if not user_input.strip():
                    continue
                
                response = self._get_mistral_response(user_input)
                self.chat_interface.display_message(response)
                
            except KeyboardInterrupt:
                self.chat_interface.display_message("\nGoodbye!")
                break
            except Exception as e:
                self.chat_interface.display_message(f"An error occurred: {str(e)}")