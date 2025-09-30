import React from 'react';
import { useState } from 'react';

function Dashboard() {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!inputValue.trim()) return;
    
    alert(`Searching for: ${inputValue}`);
    console.log('User query:', inputValue);
    setInputValue('');
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="bg-white shadow-md p-4 z-10">
        <h1 className="text-2xl font-bold text-gray-800">LitScout Dashboard</h1>
      </header>
      <main className="flex-grow p-6 overflow-y-auto">
        <div className="text-center text-gray-500">
          <p>Welcome to LitScout!</p>
          <p>Enter a topic for your literature review below to get started.</p>
        </div>
      </main>
      <footer className="p-4 bg-white border-t border-gray-200">
        <form onSubmit={handleSubmit} className="flex items-center space-x-4">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Enter your research topic, e.g., 'AI in healthcare'..."
            className="flex-grow px-4 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <button
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold p-3 rounded-full flex items-center justify-center transition duration-200"
            aria-label="Send message"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
              <path d="M3.478 2.404a.75.75 0 0 0-.926.941l2.432 7.905H13.5a.75.75 0 0 1 0 1.5H4.984l-2.432 7.905a.75.75 0 0 0 .926.94 60.519 60.519 0 0 0 18.445-8.986.75.75 0 0 0 0-1.218A60.517 60.517 0 0 0 3.478 2.404Z" />
            </svg>
          </button>
        </form>
      </footer>
    </div>
  );
}

export default Dashboard;