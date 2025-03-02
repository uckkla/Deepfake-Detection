import {BrowserRouter as Router, Route, Routes} from 'react-router-dom';
import './App.css'
import Home from "./pages/Home.jsx"
import About from "./pages/About.jsx"
import Navbar from "./components/Navbar.jsx"

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home/>} />
        <Route path="/about" element={<About/>} />
      </Routes>
    </Router>
  )
}

export default App

