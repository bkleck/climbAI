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

import RoundButton from '../components/UI/RoundButton';
import { useNavigate } from 'react-router-dom';



const Dashboard = () => {
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

  const [selectedColor, setSelectedColor] = useState('blue');

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
    fetchWeeklyStats();
  }, {});

  useEffect(() => {
    fetchColorPercentage();
  }, []);

  useEffect(() => {
    fetchCaloriesHeartRate();
  }, []);

  async function fetchNotes() {
    console.log("fetching notes")
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

  async function fetchWeeklyStats() {
    const apiData = await API.graphql({ query: listTodos });
    const notesFromAPI = apiData.data.listTodos.items;

    // instantiate weekly date dictionary
    let dates = [];
    let dateDictCalories = {};
    let dateDictCount = {};

    let date = new Date();
    let dayOfWeek = date.getDay();
    let diff = date.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1);
    let firstDayOfWeek = new Date(date.setDate(diff));

    for (let i = 0; i < 7; i++) {
      let currentDate = new Date(firstDayOfWeek);
      currentDate.setDate(currentDate.getDate() + i);
      let formattedDate = currentDate.toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "2-digit",
        year: "2-digit",
      });
      dates.push(formattedDate);
      dateDictCalories[formattedDate] = 0;
      dateDictCount[formattedDate] = 0;

      notesFromAPI.forEach(function (input) {
        if (dates.includes(input.date)) {
          dateDictCalories[input.date] += parseInt(input["timeInSeconds"]);
          dateDictCount[input.date] += 1;
        }
      });

      const finalWeeklyStats = [];

      const daysOfWeek = ["Sun", "Mon", "Tues", "Wed", "Thurs", "Fri", "Sat"];
      var idx = 0;

      console.log("dateDictCalories");
      console.log(dateDictCalories);

      for (var e of dates) {
        if (dateDictCalories[e] !== 0 && dateDictCount !== 0) {
          dateDictCalories[e] = Math.floor(
            dateDictCalories[e] / dateDictCount[e]
          );
        }

        var tempDict = {
          name: daysOfWeek[idx],
          timeTaken: dateDictCalories[e],
        };
        finalWeeklyStats.push(tempDict);
        idx += 1;
      }

      setWeeklyStats(finalWeeklyStats);
    }
  }

  async function fetchColorPercentage() {
    console.log("fetching colors")
    const apiData = await API.graphql({ query: listTodos });
    const notesFromAPI = apiData.data.listTodos.items;
    console.log(notesFromAPI)


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

    // instantiate weekly date dictionary
    let dates = [];
    let dateDictCalories = {};
    let dateDictHeartRate = {};
    let dateDictCount = {};

    let date = new Date();
    let dayOfWeek = date.getDay();
    let diff = date.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1);
    let firstDayOfWeek = new Date(date.setDate(diff));

    for (let i = 0; i < 7; i++) {
      let currentDate = new Date(firstDayOfWeek);
      currentDate.setDate(currentDate.getDate() + i);
      let formattedDate = currentDate.toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "2-digit",
        year: "2-digit",
      });
      dates.push(formattedDate);
      dateDictCalories[formattedDate] = 0;
      dateDictHeartRate[formattedDate] = 0;
      dateDictCount[formattedDate] = 0;
    }

    notesFromAPI.forEach(function (input) {
      if (dates.includes(input.date)) {
        dateDictCalories[input.date] += parseInt(input["calories"]);
        dateDictHeartRate[input.date] += parseInt(input["heartRate"]);
        dateDictCount[input.date] += 1;
      }
    });

    const finalWeeklyStats = [];

    const daysOfWeek = ["Sun", "Mon", "Tues", "Wed", "Thurs", "Fri", "Sat"];
    var idx = 0;

    for (var e of dates) {
      if (dateDictHeartRate[e] !== 0 && dateDictCount !== 0) {
        dateDictHeartRate[e] = Math.floor(
          dateDictHeartRate[e] / dateDictCount[e]
        );
      }

      var tempDict = {
        name: daysOfWeek[idx],
        calories: dateDictCalories[e],
        heartRate: dateDictHeartRate[e],
      };
      finalWeeklyStats.push(tempDict);
      idx += 1;
    }

    setCaloriesHeartRate(finalWeeklyStats);
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

  const handleClickMonth = () => {
    navigate("/dashboard-month");
  };

  const handleClickYear = () => {
    navigate("/dashboard-year");
  };


  return (
    <div className="dashboard">
      <div className="dashboard__wrapper">
        <div className="filter__widget-wrapper">
          <div className="filter__widget-01">
            <RoundButton text="Week" color="#00bfff" onClick={() => console.log("week pressed")} />
          </div>

          <div className="filter__widget-01">
            <RoundButton text="Month" color="gray" onClick={() => handleClickMonth()} />
          </div>
          <div className="filter__widget-01">
            <RoundButton text="Year" color="gray" onClick={() => handleClickYear()} />
          </div>
        </div>

        <div style={{ height: '25px' }} />

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
            <h3 className="stats__title">Total Calories Burnt (KCal) / AVG. Heart Rate (BPM)</h3>
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

export default Dashboard;
