import React from 'react';
import Layout from './Layout';
import { Link } from 'react-router-dom';

function About() {
  return (
    <Layout>
      <div className="p-8 bg-white rounded-lg shadow-md w-full max-w-3xl text-left">
        <h1 className="text-3xl font-bold text-gray-800 mb-4 text-center">About LitScout</h1>
        <section className="mb-6">
          <h2 className="text-2xl font-semibold text-gray-700 mb-2">Our Mission</h2>
          <p className="text-gray-600">
            Manually preparing a systematic literature review (SLR) takes considerable time and effort, often involving the analysis of hundreds or thousands of papers. LitScout leverages Artificial Intelligence to automate the repetitive tasks involved in this process. Our mission is to reduce the costly, error-prone, and laborious work for researchers, allowing them to focus on discovery and insight.
          </p>
        </section>
        <section>
          <h2 className="text-2xl font-semibold text-gray-700 mb-2">How It Works</h2>
          <p className="text-gray-600 mb-4">
            Our platform follows a phased approach to streamline your research workflow, with more features planned for the future. The automatic generation of content for an SLR report is a complex task that we aim to address.
          </p>
          <ul className="list-disc list-inside space-y-2 text-gray-600">
            <li>
              <strong>Phase 1: Search & Filter:</strong> You provide the initial keywords for your research, and our Search & Filter Agent connects to external databases like Semantic Scholar to retrieve an initial set of papers.
            </li>
            <li>
              <strong>Phase 2: AI-Powered Screening:</strong> Our Screening Agent uses a prompted Large Language Model (LLM) to perform an initial classification on the titles and abstracts, helping you quickly identify relevant studies.
            </li>
            <li>
              <strong>Phase 3: Analysis & Reporting (Coming Soon):</strong> Future developments will focus on in-depth data extraction and the semi-automated generation of report findings to further accelerate your research.
            </li>
          </ul>
        </section>
        <div className="text-center mt-8">
            <Link 
                to="/" 
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition duration-200"
            >
                Get Started
            </Link>
        </div>
      </div>
    </Layout>
  );
}

export default About;