import { BrowserRouter as Router, Route, Link } from "react-router-dom";
import React from "react";
import {ContentRoute} from "../General/ContentRoute";


export default ({routes}) => {

    return (
        <div>
            Index
            {routes.map((route, i) => (
                <ContentRoute key={i} {...route} />
            ))}
        </div>
    );
}