import "../styles/dashboard.css";
import SingleCard from "../components/reuseable/SingleCard";

import blueRockWall from "../assets/images/blue-rockwall.png";
import greenRockWall from "../assets/images/green-rockwall.png";
import redRockWall from "../assets/images/red-rockwall.png";

import MileChart from "../charts/MileChart";
import CarStatsChart from "../charts/CarStatsChart";
import RecommendCarCard from "../components/UI/RecommendCarCard";

import recommendCarsData from "../assets/dummy-data/recommendCars";
import React, { useState, useEffect } from "react";

import { API } from "aws-amplify";
import { listTodos } from "../graphql/queries";
import { createTodo, deleteTodo } from "../graphql/mutations";

import RoundButton from "../components/UI/RoundButton";
import { useNavigate } from "react-router-dom";

const DashboardMonth = () => {
  const carObj = {
    title: "Total Climbs",
    totalNumber: 750,
    icon: "ri-numbers-line",
  };

  const tripObj = {
    title: "Total Hours Spent",
    totalNumber: 1697,
    icon: "ri-time-line",
  };

  const clientObj = {
    title: "Total KCalories Burnt",
    totalNumber: "85k",
    icon: "ri-timer-flash-line",
  };

  const distanceObj = {
    title: "AVG. Heart Rate",
    totalNumber: 180,
    icon: "ri-heart-pulse-fill",
  };

  const [selectedColor, setSelectedColor] = useState("blue");

  const [notes, setNotes] = useState([]);
  const [climbs, setClimbs] = useState({});
  const [minutes, setMinutes] = useState({});
  const [calories, setCalories] = useState({});
  const [heartRate, setHeartRate] = useState({});
  const [weeklyStats, setWeeklyStats] = useState([]);
  const [colorPercentage, setColorPercentage] = useState([]);
  const [caloriesHeartRate, setCaloriesHeartRate] = useState([]);

  useEffect(() => {
    fetchNotes();
  }, []);

  useEffect(() => {
    fetchClimbs();
  }, {});

  useEffect(() => {
    fetchMinutes();
  }, {});

  useEffect(() => {
    fetchCalories();
  }, {});

  useEffect(() => {
    fetchHeartRate();
  }, {});

  useEffect(() => {
    fetchMonthlyStats();
  }, {});

  useEffect(() => {
    fetchColorPercentage();
  }, []);

  useEffect(() => {
    fetchCaloriesHeartRate();
  }, []);

  async function fetchNotes() {
    const apiData = await API.graphql({ query: listTodos });
    const notesFromAPI = apiData.data.listTodos.items;
    setNotes(notesFromAPI);
    console.log(notesFromAPI);
  }

  async function fetchClimbs() {
    const apiData = await API.graphql({ query: listTodos });
    const notesFromAPI = apiData.data.listTodos.items;
    const numClimbs = notesFromAPI.length;

    const climbObject = {
      title: "Total Climbs",
      totalNumber: numClimbs.toString(),
      icon: "ri-numbers-line",
    };
    setClimbs(climbObject);
  }

  async function fetchMinutes() {
    const apiData = await API.graphql({ query: listTodos });
    const notesFromAPI = apiData.data.listTodos.items;
    const numSeconds = notesFromAPI.reduce((acc, curr) => {
      return acc + parseInt(curr["timeInSeconds"]);
    }, 0);

    const minutesObject = {
      title: "Total Minutes Spent",
      totalNumber: Math.floor(numSeconds / 60).toString(),
      icon: "ri-time-line",
    };
    setMinutes(minutesObject);
  }

  async function fetchCalories() {
    const apiData = await API.graphql({ query: listTodos });
    const notesFromAPI = apiData.data.listTodos.items;
    const numCalories = notesFromAPI.reduce((acc, curr) => {
      return acc + parseInt(curr["calories"]);
    }, 0);

    const caloriesObject = {
      title: "Total KCalories Burnt",
      totalNumber: numCalories.toString(),
      icon: "ri-timer-flash-line",
    };
    setCalories(caloriesObject);
  }

  async function fetchHeartRate() {
    const apiData = await API.graphql({ query: listTodos });
    const notesFromAPI = apiData.data.listTodos.items;
    const length = notesFromAPI.length;
    const totalHeartRate = notesFromAPI.reduce((acc, curr) => {
      return acc + parseInt(curr["heartRate"]);
    }, 0);

    const heartRateObject = {
      title: "AVG. Heart Rate",
      totalNumber: Math.floor(totalHeartRate / length).toString(),
      icon: "ri-heart-pulse-fill",
    };
    setHeartRate(heartRateObject);
  }

  async function fetchMonthlyStats() {
    const apiData = await API.graphql({ query: listTodos });
    const notesFromAPI = apiData.data.listTodos.items;

    let dates = [0, 1, 2, 3];
    let dateDictCalories = {};
    let dateDictCount = {};

    dateDictCalories[0] = 0;
    dateDictCalories[1] = 0;
    dateDictCalories[2] = 0;
    dateDictCalories[3] = 0;

    dateDictCount[0] = 0;
    dateDictCount[1] = 0;
    dateDictCount[2] = 0;
    dateDictCount[3] = 0;

    var date = new Date();
    var month = date.getMonth();
    var year = date.getFullYear();

    var firstDay = new Date(year, month, 1);
    var lastDay = new Date(year, month + 1, 0);

    var quarter1Start = new Date(year, month, 1);
    var quarter1End = new Date(
      year,
      month,
      Math.floor((lastDay.getDate() - firstDay.getDate()) / 4) +
        firstDay.getDate()
    );

    var quarter2Start = new Date(year, month, quarter1End.getDate() + 1);
    var quarter2End = new Date(
      year,
      month,
      quarter1End.getDate() +
        Math.floor((lastDay.getDate() - quarter1End.getDate()) / 3)
    );

    var quarter3Start = new Date(year, month, quarter2End.getDate() + 1);
    var quarter3End = new Date(
      year,
      month,
      quarter2End.getDate() +
        Math.floor((lastDay.getDate() - quarter2End.getDate()) / 2)
    );

    var quarter4Start = new Date(year, month, quarter3End.getDate() + 1);
    var quarter4End = lastDay;

    notesFromAPI.forEach(function (input) {
      var dateComponents = input.date.split("/");
      var day = parseInt(dateComponents[0]);
      var month = parseInt(dateComponents[1]) - 1;
      var year = 2000 + parseInt(dateComponents[2]);
      var date = new Date(year, month, day);

      if (date >= quarter1Start && date <= quarter1End) {
        dateDictCalories[0] += parseInt(input["timeInSeconds"]);
        dateDictCount[0] += 1;
      } else if (date >= quarter2Start && date <= quarter2End) {
        dateDictCalories[1] += parseInt(input["timeInSeconds"]);
        dateDictCount[1] += 1;
      } else if (date >= quarter3Start && date <= quarter3End) {
        dateDictCalories[2] += parseInt(input["timeInSeconds"]);
        dateDictCount[2] += 1;
      } else if (date >= quarter4Start && date <= quarter4End) {
        dateDictCalories[3] += parseInt(input["timeInSeconds"]);
        dateDictCount[3] += 1;
      }
    });

    const finalMonthlyStats = [];

    console.log("monthly stats");
    console.log(dateDictCalories);
    console.log(dateDictCount);

    for (var e of dates) {
      if (dateDictCalories[e] !== 0 && dateDictCount[e] !== 0) {
        dateDictCalories[e] = Math.floor(
          dateDictCalories[e] / dateDictCount[e]
        );
        
      }

      var tempDict = {
        name: "Week " + (e + 1).toString(),
        timeTaken: dateDictCalories[e],
      };
      finalMonthlyStats.push(tempDict);
    }

    setWeeklyStats(finalMonthlyStats);
  }

  async function fetchColorPercentage() {
    const apiData = await API.graphql({ query: listTodos });
    const notesFromAPI = apiData.data.listTodos.items;

    const length = notesFromAPI.length;

    let colorsPercentage = {};
    colorsPercentage["blue"] = 0;
    colorsPercentage["green"] = 0;
    colorsPercentage["red"] = 0;

    notesFromAPI.forEach(function (input) {
      colorsPercentage[input.color] += 1;
    });

    colorsPercentage["blue"] = Math.floor(
      (100 * colorsPercentage["blue"]) / length
    );
    colorsPercentage["green"] = Math.floor(
      (100 * colorsPercentage["green"]) / length
    );
    colorsPercentage["red"] = Math.floor(
      (100 * colorsPercentage["red"]) / length
    );

    const colorsObject = [
      {
        id: "01",
        carName: "Mini Cooper",
        rentPrice: 32,
        retweet: "132",
        imgUrl: blueRockWall,
        percentage: colorsPercentage["blue"].toString(),
      },
      {
        id: "02",
        carName: "Porsche 911 Carrera",
        rentPrice: 28,
        retweet: "130",
        imgUrl: greenRockWall,
        percentage: colorsPercentage["green"].toString(),
      },
      {
        id: "03",
        carName: "Porsche 911 Carrera",
        rentPrice: 28,
        retweet: "130",
        imgUrl: redRockWall,
        percentage: colorsPercentage["red"].toString(),
      },
    ];

    setColorPercentage(colorsObject);
  }

  async function fetchCaloriesHeartRate() {
    const apiData = await API.graphql({ query: listTodos });
    const notesFromAPI = apiData.data.listTodos.items;

    let dates = [0, 1, 2, 3];
    let dateDictCalories = {};
    let dateDictHeartRate = {};
    let dateDictCount = {};

    dateDictCalories[0] = 0;
    dateDictCalories[1] = 0;
    dateDictCalories[2] = 0;
    dateDictCalories[3] = 0;

    dateDictHeartRate[0] = 0;
    dateDictHeartRate[1] = 0;
    dateDictHeartRate[2] = 0;
    dateDictHeartRate[3] = 0;

    dateDictCount[0] = 0;
    dateDictCount[1] = 0;
    dateDictCount[2] = 0;
    dateDictCount[3] = 0;

    var date = new Date();
    var month = date.getMonth();
    var year = date.getFullYear();

    var firstDay = new Date(year, month, 1);
    var lastDay = new Date(year, month + 1, 0);

    var quarter1Start = new Date(year, month, 1);
    var quarter1End = new Date(
      year,
      month,
      Math.floor((lastDay.getDate() - firstDay.getDate()) / 4) +
        firstDay.getDate()
    );

    var quarter2Start = new Date(year, month, quarter1End.getDate() + 1);
    var quarter2End = new Date(
      year,
      month,
      quarter1End.getDate() +
        Math.floor((lastDay.getDate() - quarter1End.getDate()) / 3)
    );

    var quarter3Start = new Date(year, month, quarter2End.getDate() + 1);
    var quarter3End = new Date(
      year,
      month,
      quarter2End.getDate() +
        Math.floor((lastDay.getDate() - quarter2End.getDate()) / 2)
    );

    var quarter4Start = new Date(year, month, quarter3End.getDate() + 1);
    var quarter4End = lastDay;

    notesFromAPI.forEach(function (input) {
      var dateComponents = input.date.split("/");
      var day = parseInt(dateComponents[0]);
      var month = parseInt(dateComponents[1]) - 1;
      var year = 2000 + parseInt(dateComponents[2]);
      var date = new Date(year, month, day);

      if (date >= quarter1Start && date <= quarter1End) {
        dateDictCalories[0] += parseInt(input["calories"]);
        dateDictHeartRate[0] += parseInt(input["heartRate"]);
        dateDictCount[0] += 1;

      } else if (date >= quarter2Start && date <= quarter2End) {
        dateDictCalories[1] += parseInt(input["calories"]);
        dateDictHeartRate[1] += parseInt(input["heartRate"]);
        dateDictCount[1] += 1;

      } else if (date >= quarter3Start && date <= quarter3End) {
        dateDictCalories[2] += parseInt(input["calories"]);
        dateDictHeartRate[2] += parseInt(input["heartRate"]);
        dateDictCount[2] += 1;

      } else if (date >= quarter4Start && date <= quarter4End) {
        dateDictCalories[3] += parseInt(input["calories"]);
        dateDictHeartRate[3] += parseInt(input["heartRate"]);
        dateDictCount[3] += 1;

      }
    });

    const finalMonthlyStats = [];

    console.log("monthly stats");
    console.log(dateDictCalories);
    console.log(dateDictCount);

    for (var e of dates) {
      if (dateDictCalories[e] !== 0 && dateDictCount[e] !== 0) {
        console.log("after divide");
        console.log(dateDictCalories[e]);
        console.log(dateDictCount[e]);
        dateDictCalories[e] = Math.floor(
          dateDictCalories[e] / dateDictCount[e]
        );
        
      }

      var tempDict = {
        name: "Week " + (e + 1).toString(),
        calories: dateDictCalories[e],
        heartRate: dateDictHeartRate[e],
      };
      finalMonthlyStats.push(tempDict);
    }

    setCaloriesHeartRate(finalMonthlyStats);

  }

  async function createNote(event) {
    event.preventDefault();
    const form = new FormData(event.target);
    const data = {
      name: form.get("name"),
      description: form.get("description"),
    };
    await API.graphql({
      query: createTodo,
      variables: { input: data },
    });
    fetchNotes();
    event.target.reset();
  }

  const navigate = useNavigate();

  const handleClickWeek = () => {
    navigate("/dashboard");
  };

  const handleClickYear = () => {
    navigate("/dashboard-year");
  };

  return (
    <div className="dashboard">
      <div className="dashboard__wrapper">
        <div className="filter__widget-wrapper">
          <div className="filter__widget-01">
            <RoundButton
              text="Week"
              color="gray"
              onClick={() => handleClickWeek()}
            />
          </div>

          <div className="filter__widget-01">
            <RoundButton
              text="Month"
              color="#00bfff"
              onClick={() => console.log("month pressed")}
            />
          </div>
          <div className="filter__widget-01">
            <RoundButton
              text="Year"
              color="gray"
              onClick={() => handleClickYear()}
            />
          </div>
        </div>

        <div style={{ height: "25px" }} />

        <div className="dashboard__cards">
          <SingleCard item={climbs} />
          <SingleCard item={minutes} />
          <SingleCard item={calories} />
          <SingleCard item={heartRate} />
        </div>

        <div className="statics">
          <div className="stats">
            <h3 className="stats__title">AVG. Time Taken (s)</h3>
            <MileChart mileStaticsData={weeklyStats} />
          </div>

          <div className="stats">
            <h3 className="stats__title">
              Total Calories Burnt (KCal) / AVG. Heart Rate (BPM)
            </h3>
            <CarStatsChart carStaticsData={caloriesHeartRate} />
          </div>
        </div>

        <div className="recommend__cars-wrapper">
          {colorPercentage.map((item) => (
            <RecommendCarCard item={item} key={item.id} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default DashboardMonth;
