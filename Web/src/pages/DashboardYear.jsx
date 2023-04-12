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

const DashboardYear = () => {
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

    let dates = [0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11];
    let dateDictCalories = {};
    let dateDictHeartRate = {};
    let dateDictCount = {};
    var shortMonthNames = {
      0: "Jan",
      1: "Feb",
      2: "Mar",
      3: "Apr",
      4: "May",
      5: "Jun",
      6: "Jul",
      7: "Aug",
      8: "Sep",
      9: "Oct",
      10: "Nov",
      11: "Dec"
    };

    dateDictCalories[0] = 0;
    dateDictCalories[1] = 0;
    dateDictCalories[2] = 0;
    dateDictCalories[3] = 0;
    dateDictCalories[4] = 0;
    dateDictCalories[5] = 0;
    dateDictCalories[6] = 0;
    dateDictCalories[7] = 0;
    dateDictCalories[8] = 0;
    dateDictCalories[9] = 0;
    dateDictCalories[10] = 0;
    dateDictCalories[11] = 0;

    dateDictCount[0] = 0;
    dateDictCount[1] = 0;
    dateDictCount[2] = 0;
    dateDictCount[3] = 0;
    dateDictCount[4] = 0;
    dateDictCount[5] = 0;
    dateDictCount[6] = 0;
    dateDictCount[7] = 0;
    dateDictCount[8] = 0;
    dateDictCount[9] = 0;
    dateDictCount[10] = 0;
    dateDictCount[11] = 0;

    var date = new Date();
    var current_year = date.getFullYear();

    notesFromAPI.forEach(function (input) {
      var dateComponents = input.date.split("/");
      var day = parseInt(dateComponents[0]);
      var month = parseInt(dateComponents[1]) - 1;
      var year = 2000 + parseInt(dateComponents[2]);
      var date = new Date(year, month, day);

      if (current_year==year){
        var index = month
        dateDictCalories[index] += parseInt(input["timeInSeconds"]);
        dateDictCount[index] += 1;
      }
    });

    const finalYearlyStats = [];


    for (var e of dates) {
      if (dateDictCalories[e] !== 0 && dateDictCount[e] !== 0) {
        dateDictCalories[e] = Math.floor(
          dateDictCalories[e] / dateDictCount[e]
        );
        
      }

      var tempDict = {
        name: shortMonthNames[e],
        timeTaken: dateDictCalories[e],
      };
      finalYearlyStats.push(tempDict);
    }

    setWeeklyStats(finalYearlyStats);
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

    let dates = [0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11];
    let dateDictCalories = {};
    let dateDictHeartRate = {};
    let dateDictCount = {};

    dateDictCalories[0] = 0;
    dateDictCalories[1] = 0;
    dateDictCalories[2] = 0;
    dateDictCalories[3] = 0;
    dateDictCalories[4] = 0;
    dateDictCalories[5] = 0;
    dateDictCalories[6] = 0;
    dateDictCalories[7] = 0;
    dateDictCalories[8] = 0;
    dateDictCalories[9] = 0;
    dateDictCalories[10] = 0;
    dateDictCalories[11] = 0;

    dateDictHeartRate[0] = 0;
    dateDictHeartRate[1] = 0;
    dateDictHeartRate[2] = 0;
    dateDictHeartRate[3] = 0;
    dateDictHeartRate[4] = 0;
    dateDictHeartRate[5] = 0;
    dateDictHeartRate[6] = 0;
    dateDictHeartRate[7] = 0;
    dateDictHeartRate[8] = 0;
    dateDictHeartRate[9] = 0;
    dateDictHeartRate[10] = 0;
    dateDictHeartRate[11] = 0;

    dateDictCount[0] = 0;
    dateDictCount[1] = 0;
    dateDictCount[2] = 0;
    dateDictCount[3] = 0;
    dateDictCount[4] = 0;
    dateDictCount[5] = 0;
    dateDictCount[6] = 0;
    dateDictCount[7] = 0;
    dateDictCount[8] = 0;
    dateDictCount[9] = 0;
    dateDictCount[10] = 0;
    dateDictCount[11] = 0;

    var shortMonthNames = {
      0: "Jan",
      1: "Feb",
      2: "Mar",
      3: "Apr",
      4: "May",
      5: "Jun",
      6: "Jul",
      7: "Aug",
      8: "Sep",
      9: "Oct",
      10: "Nov",
      11: "Dec"
    };

    var date = new Date();
    var current_year = date.getFullYear();

    notesFromAPI.forEach(function (input) {
      var dateComponents = input.date.split("/");
      var day = parseInt(dateComponents[0]);
      var month = parseInt(dateComponents[1]) - 1;
      var year = 2000 + parseInt(dateComponents[2]);
      var date = new Date(year, month, day);

      if (current_year==year){
        var index = month
        dateDictCalories[index] += parseInt(input["calories"]);
        dateDictHeartRate[index] += parseInt(input["heartRate"]);
        dateDictCount[index] += 1;
      }
    });

    const finalYearlyStats = [];


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
        name: shortMonthNames[e],
        calories: dateDictCalories[e],
        heartRate: dateDictHeartRate[e],
      };
      finalYearlyStats.push(tempDict);
    }

    setCaloriesHeartRate(finalYearlyStats);

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

  const handleClickMonth = () => {
    navigate("/dashboard-month");
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
              color="gray"
              onClick={() => handleClickMonth()}
            />
          </div>
          <div className="filter__widget-01">
            <RoundButton
              text="Year"
              color="#00bfff"
              onClick={() => console.log("year clicked")}
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

export default DashboardYear;
