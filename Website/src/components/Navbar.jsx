import React from "react";
import { Link } from "react-router-dom";
import './Navbar.css'

const Navbar = () => {
    return (
        <nav className="navbar">
            <div className="navbar-left">
                <Link to="/" className="logo">
                    Home
                </Link>
            </div>
            <div className="navbar-right">
                <Link to="/about" className="about">
                    About
                </Link>
            </div>
        </nav>
    );
};

export default Navbar;