import React from "react";
import { BrowserRouter as Router, Route, Link } from "react-router-dom";


function ContentRoute(route) {
    console.log(route.path);
    return (
        <Route
            path={route.path}
            render={props => (
                // pass the sub-routes down to keep nesting
                <route.component {...props} routes={route.routes}/>
            )}
            exact
        />
    );
}

export {ContentRoute};