import React from "react";

import { ResponsiveContainer, BarChart, Bar, XAxis, Tooltip } from "recharts";

import mileStaticsData from "../assets/dummy-data/mileStatics";

const mileChart = (props) => {
  return (
    <ResponsiveContainer width="100%">
      <BarChart data={props.mileStaticsData}>
        <XAxis dataKey="name" stroke="#2884ff" />
        <Bar dataKey="timeTaken" stroke="#2884ff" fill="#2884ff" barSize={30} />

        <Tooltip wrapperClassName="tooltip__style" cursor={false} />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default mileChart;
