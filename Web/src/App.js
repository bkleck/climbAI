import "./App.css";
import Layout from "./components/Layout/Layout";
import "@aws-amplify/ui-react/styles.css";
import {
  withAuthenticator,
  Button,
  Heading,
  Image,
  View,
  Card,
} from "@aws-amplify/ui-react";

function App() {
  return <Layout />;
}

// export default withAuthenticator(App);
export default App;
