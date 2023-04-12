import React from "react";

import { Link } from "react-router-dom";
import profileImg from "../../assets/images/profile-02.png";
import "./top-nav.css";

const TopNav = () => {
  return (
    <div className="top__nav">
      <div className="top__nav-wrapper">
        <div className="notification">
          <span>
            <i></i>
          </span>
        </div>
        <div className="top__nav-right">
          <span className="notification">
            <i class="ri-award-fill"></i>
            {/* <span className="badge">1</span> */}
          </span>
          <div className="profile">
            {/* <Link to="/settings"> */}
            <img src={profileImg} alt="" />
            {/* </Link> */}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TopNav;
