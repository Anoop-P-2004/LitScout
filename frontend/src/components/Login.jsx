import React from 'react'; // <-- This is the fix
import { Link, useNavigate } from 'react-router-dom';
import Layout from './Layout';
import { useState } from 'react';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (event) => {
    event.preventDefault();
    console.log('Login attempt with:', { email, password });
    navigate('/dashboard');
  };

  
  return (
    <Layout>
      <div className="p-8 bg-white rounded-lg shadow-md w-full max-w-sm">
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-gray-800">LitScout</h1>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="email" className="block text-gray-700 font-semibold mb-2">
              Email
            </label>
            <input 
              type="email" 
              id="email" 
              name="email"
              placeholder="you@example.com"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring focus:ring-blue-200"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="mb-4">
            <div className="flex justify-between items-center">
              <label htmlFor="password" className="block text-gray-700 font-semibold">
                Password
              </label>
              <a href="#" className="text-sm text-blue-600 hover:underline">
                Forgot Password?
              </a>
            </div>
            <input 
              type="password" 
              id="password" 
              name="password"
              placeholder="••••••••"
              className="w-full mt-2 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring focus:ring-blue-200"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <button 
            type="submit"
            className="w-full py-2 mt-4 px-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition duration-200"
          >
            Login
          </button>
        </form>
        <p className="text-center text-sm text-gray-600 mt-6">
          Don't have an account?{' '}
          <Link to="/signup" className="font-semibold text-blue-600 hover:underline">
            Sign Up
          </Link>
        </p>
      </div>
    </Layout>
  );
}

export default Login;