import React from "react";

const RoundButton = ({text, color, onClick }) => (
  <div
    style={{
      backgroundColor: color,
      color: "white",
      border: "1px solid gray",
      borderRadius: "5px",
      width: "80px",
      height: "30px",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      cursor: "pointer",
    }}
    onClick={onClick}
  >
    <span>{text}</span>
  </div>
);

export default RoundButton;
