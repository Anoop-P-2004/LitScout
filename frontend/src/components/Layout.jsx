import React from 'react';
import { Link } from 'react-router-dom';

function Layout({ children }) {
  return (
    <div 
      className="relative min-h-screen bg-cover bg-center"
      style={{ backgroundImage: "url('https://images.unsplash.com/photo-1481627834876-b7833e8f5570?q=80&w=2128')" }}
    >
      <div className="absolute inset-0 bg-black opacity-50"></div>
      <main className="relative z-10 flex items-center justify-center min-h-screen">
        {children}
      </main>
      <footer className="absolute bottom-4 text-center w-full text-xs text-white z-10">
        <div className="mb-2">
          <Link to="/about" className="mx-2 hover:underline">About</Link>
          <a href="#" className="mx-2 hover:underline">Contact</a>
          <a href="#" className="mx-2 hover:underline">Terms of Service</a>
        </div>
        <div>
          &copy; {new Date().getFullYear()} LitScout. All Rights Reserved.
        </div>
      </footer>
    </div>
  );
}

export default Layout;